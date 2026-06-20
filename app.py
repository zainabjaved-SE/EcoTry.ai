from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import uuid
import cv2
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, session, url_for)
from gradio_client import Client, file
from PIL import Image

from ai_stylist import detect_skin_tone_pil, generate_style_text_suggestions
from live_tryon import apply_cloth_to_photo
from utils import get_body_measurements
from size_model import recommend_size

# ── Config ────────────────────────────────────────────────────────────────────
DATA_CSV       = Path("eco_try_products_dataset_fabric_category_fixed.csv")
OVERRIDES_JSON = Path("overrides.json")

VTON_SPACE_ID = os.environ.get("VTON_SPACE_ID", "EcoTry/IDM-VTON").strip()
VTON_API_NAME = os.environ.get("VTON_API_NAME", "/tryon").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN")


_ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_VTON_CLIENT: Optional[Client] = None


@dataclass(frozen=True)
class Product:
    product_id: int
    product_name: str
    fabric_type: str
    image_url: str
    water_usage_liters: float
    co2_emission_kg: float
    biodegradability_score: float
    sustainability_score: float
    awareness_text: str
    category: str
    static_filename: str


def infer_category(product_name: str) -> str:
    if not isinstance(product_name, str) or not product_name.strip():
        return "Other"
    parts = re.sub(r"[^A-Za-z]+", " ", product_name).strip().split()
    if not parts:
        return "Other"
    last = parts[-1].lower()
    mapping = {
        "tshirt": "T-Shirt", "tee": "T-Shirt", "shirt": "Shirt",
        "jacket": "Jacket",  "hoodie": "Hoodie", "sweater": "Sweater",
        "jeans": "Jeans",    "pants": "Pants",   "trouser": "Pants",
        "trousers": "Pants", "shorts": "Shorts", "dress": "Dress",
        "skirt": "Skirt",    "coat": "Coat",
    }
    return mapping.get(last, last.capitalize())


def normalize_static_image_path(image_url_value: Any) -> str:
    if not isinstance(image_url_value, str) or not image_url_value.strip():
        return "images/placeholder.png"
    s = image_url_value.strip().replace("\\", "/").lstrip("/")
    if s.startswith("static/"):
        s = s[len("static/"):]
    return s


def load_overrides() -> Dict[str, Dict[str, Any]]:
    if not OVERRIDES_JSON.exists():
        OVERRIDES_JSON.write_text("{}", encoding="utf-8")
        return {}
    try:
        return json.loads(OVERRIDES_JSON.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def save_overrides(overrides: Dict[str, Dict[str, Any]]) -> None:
    OVERRIDES_JSON.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_overrides(p: Product, overrides: Dict[str, Dict[str, Any]]) -> Product:
    override = overrides.get(str(p.product_id))
    if not override:
        return p
    allowed = {"product_name", "fabric_type", "category"}
    data = {k: v for k, v in override.items()
            if k in allowed and isinstance(v, str) and v.strip()}
    if not data:
        return p
    return replace(p,
        product_name=data.get("product_name", p.product_name),
        fabric_type=data.get("fabric_type",   p.fabric_type),
        category=data.get("category",         p.category))


def _get_vton_client() -> Client:
    global _VTON_CLIENT
    if _VTON_CLIENT is not None:
        return _VTON_CLIENT
    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)
    print(f"[VTON] Connecting to {VTON_SPACE_ID}...")
    _VTON_CLIENT = Client(src=VTON_SPACE_ID)
    print("[VTON] Connected!")
    return _VTON_CLIENT


def _secure_ext(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    return ext if ext in _ALLOWED_IMAGE_EXTS else ".png"


def _open_result_as_image(result: Any) -> Image.Image:
    if isinstance(result, Image.Image):
        return result.convert("RGB")
    if isinstance(result, str):
        return Image.open(result).convert("RGB")
    if isinstance(result, dict) and "path" in result:
        return Image.open(result["path"]).convert("RGB")
    raise ValueError(f"Unexpected VTON result: {type(result)}")


def call_vton_space(person_image_path: str,
                    cloth_image_path: str,
                    garment_description: str) -> Image.Image:
    global _VTON_CLIENT
    try:
        client = _get_vton_client()
        print("[VTON] Sending to IDM-VTON...")
        result = client.predict(
            dict={
                "background": file(person_image_path),
                "layers": [],
                "composite": None,
            },
            garm_img=file(cloth_image_path),
            garment_des=f"high quality realistic photo of person wearing {garment_description}",
            is_checked=True,
            is_checked_crop=True,
            denoise_steps=40,
            seed=1234,
            api_name=VTON_API_NAME,
        )
        print("[VTON] Success!")
        return _open_result_as_image(result[0])
    except Exception as e:
        _VTON_CLIENT = None
        print(f"[VTON] Failed: {e}")
        raise RuntimeError(f"IDM-VTON failed: {e}")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"]     = os.environ.get("ECOTRY_SECRET_KEY", "change-this-secret")
    app.config["ADMIN_PASSWORD"] = os.environ.get("ECOTRY_ADMIN_PASSWORD", "admin123")

    df = pd.read_csv(DATA_CSV)
    df["category"]        = df["product_name"].apply(infer_category)
    df["static_filename"] = df["image_url"].apply(normalize_static_image_path)

    base_products: List[Product] = []
    for row in df.to_dict(orient="records"):
        base_products.append(Product(
            product_id=int(row["product_id"]),
            product_name=str(row["product_name"]),
            fabric_type=str(row["fabric_type"]),
            image_url=str(row["image_url"]),
            water_usage_liters=float(row["water_usage_liters"]),
            co2_emission_kg=float(row["co2_emission_kg"]),
            biodegradability_score=float(row["biodegradability_score"]),
            sustainability_score=float(row["sustainability_score"]),
            awareness_text=str(row["awareness_text"]),
            category=str(row["category"]),
            static_filename=str(row["static_filename"]),
        ))

    base_by_id: Dict[int, Product] = {p.product_id: p for p in base_products}

    def get_cart() -> Dict[str, int]:
        cart = session.get("cart", {})
        if not isinstance(cart, dict):
            return {}
        cleaned: Dict[str, int] = {}
        for k, v in cart.items():
            try:
                qty = int(v)
            except (TypeError, ValueError):
                continue
            if qty > 0:
                cleaned[str(k)] = qty
        return cleaned

    def save_cart(cart: Dict[str, int]) -> None:
        session["cart"] = cart
        session.modified = True

    def cart_count(cart: Dict[str, int]) -> int:
        return sum(cart.values())

    def build_category_counts(items: List[Product]) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        for p in items:
            counts[p.category] = counts.get(p.category, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    def apply_filters(items: List[Product], *, category: Optional[str],
                      q: Optional[str], sort: str) -> List[Product]:
        filtered = items
        if category and category != "All":
            filtered = [p for p in filtered if p.category == category]
        if q:
            query = q.strip().lower()
            if query:
                filtered = [p for p in filtered
                            if query in p.product_name.lower()
                            or query in p.fabric_type.lower()]
        if sort == "eco_desc":
            filtered = sorted(filtered, key=lambda p: p.sustainability_score, reverse=True)
        elif sort == "eco_asc":
            filtered = sorted(filtered, key=lambda p: p.sustainability_score)
        elif sort == "name_desc":
            filtered = sorted(filtered, key=lambda p: p.product_name.lower(), reverse=True)
        else:
            filtered = sorted(filtered, key=lambda p: p.product_name.lower())
        return filtered

    def get_products_with_overrides() -> List[Product]:
        overrides = load_overrides()
        return [apply_overrides(p, overrides) for p in base_products]

    def render_tryon_page(product: Product, *, result_image_url: Optional[str] = None) -> str:
        return render_template("tryon.html", product={
            "product_id":   product.product_id,
            "product_name": product.product_name,
            "image_src":    url_for("static", filename=product.static_filename),
        }, result_image_url=result_image_url)

    @app.context_processor
    def inject_globals() -> Dict[str, Any]:
        cart = get_cart()
        return {
            "cart_items_count": cart_count(cart),
            "brand_logo": url_for("static", filename="assets/ecotry-logo.png"),
        }

    @app.route("/")
    def home():
        selected_category = request.args.get("category", "All")
        q    = request.args.get("q", "")
        sort = request.args.get("sort", "name_asc")
        products = get_products_with_overrides()
        filtered = apply_filters(products, category=selected_category, q=q, sort=sort)
        view_products: List[Dict[str, Any]] = []
        for p in filtered:
            view_products.append({
                "product_id": p.product_id, "product_name": p.product_name,
                "fabric_type": p.fabric_type,
                "sustainability_score": p.sustainability_score,
                "water_usage_liters": p.water_usage_liters,
                "co2_emission_kg": p.co2_emission_kg,
                "biodegradability_score": p.biodegradability_score,
                "awareness_text": p.awareness_text, "category": p.category,
                "image_src": url_for("static", filename=p.static_filename),
            })
        return render_template("index.html", products=view_products,
            categories_with_counts=build_category_counts(products),
            total_count=len(products),
            selected_category=selected_category, q=q, sort=sort)

    @app.route("/add-to-cart/<int:product_id>", methods=["POST"])
    def add_to_cart(product_id: int):
        if product_id not in base_by_id:
            flash("Product not found.", "error")
            return redirect(url_for("home"))
        cart = get_cart()
        key = str(product_id)
        cart[key] = cart.get(key, 0) + 1
        save_cart(cart)
        flash("Added to cart.", "success")
        return redirect(request.referrer or url_for("home"))

    @app.route("/cart")
    def cart_page():
        cart = get_cart()
        overrides = load_overrides()
        lines: List[Dict[str, Any]] = []
        for pid_str, qty in cart.items():
            p = base_by_id.get(int(pid_str))
            if not p:
                continue
            p = apply_overrides(p, overrides)
            lines.append({
                "product_id": p.product_id, "product_name": p.product_name,
                "fabric_type": p.fabric_type, "category": p.category,
                "qty": qty, "eco": p.sustainability_score,
                "image_src": url_for("static", filename=p.static_filename),
            })
        return render_template("cart.html", lines=lines)

    @app.route("/cart/update", methods=["POST"])
    def cart_update():
        cart = get_cart()
        for k, v in request.form.to_dict().items():
            if not k.startswith("qty_"):
                continue
            pid = k.replace("qty_", "").strip()
            try:
                qty = int(v)
            except ValueError:
                qty = 1
            if qty <= 0:
                cart.pop(pid, None)
            else:
                cart[pid] = min(qty, 99)
        save_cart(cart)
        flash("Cart updated.", "success")
        return redirect(url_for("cart_page"))

    @app.route("/cart/remove/<int:product_id>", methods=["POST"])
    def cart_remove(product_id: int):
        cart = get_cart()
        cart.pop(str(product_id), None)
        save_cart(cart)
        flash("Removed from cart.", "success")
        return redirect(url_for("cart_page"))

    @app.route("/checkout", methods=["GET", "POST"])
    def checkout():
        cart = get_cart()
        if not cart:
            flash("Your cart is empty.", "error")
            return redirect(url_for("home"))
        overrides = load_overrides()
        lines: List[Dict[str, Any]] = []
        for pid_str, qty in cart.items():
            p = base_by_id.get(int(pid_str))
            if not p:
                continue
            p = apply_overrides(p, overrides)
            lines.append({"product_id": p.product_id,
                          "product_name": p.product_name,
                          "qty": qty, "eco": p.sustainability_score})
        eco_avg = round(
            sum(l["eco"] * l["qty"] for l in lines)
            / max(1, sum(l["qty"] for l in lines)), 2)
        if request.method == "POST":
            name    = request.form.get("name",    "").strip()
            email   = request.form.get("email",   "").strip()
            address = request.form.get("address", "").strip()
            if not name or not email or not address:
                flash("Please fill in name, email, and address.", "error")
                return render_template("checkout.html", lines=lines, eco_avg=eco_avg)
            session["last_order"] = {
                "name": name, "email": email, "address": address,
                "items": lines, "eco_avg": eco_avg}
            save_cart({})
            return redirect(url_for("order_success"))
        return render_template("checkout.html", lines=lines, eco_avg=eco_avg)

    @app.route("/order-success")
    def order_success():
        order = session.get("last_order")
        if not order:
            return redirect(url_for("home"))
        return render_template("success.html", order=order)

    @app.route("/stylist", methods=["GET", "POST"])
    def stylist_page():
        if request.method == "GET":
            return render_template("stylist.html", result=None)
        selfie = request.files.get("selfie")
        if not selfie or not selfie.filename:
            flash("Please upload an image.", "error")
            return render_template("stylist.html", result=None)
        height_cm_raw = (request.form.get("height_cm") or "").strip()
        body_type     = (request.form.get("body_type") or "Average").strip()
        gender        = (request.form.get("gender")    or "Unspecified").strip()
        height_cm: Optional[float] = None
        if height_cm_raw:
            try:
                height_cm = float(height_cm_raw)
            except ValueError:
                height_cm = None
        try:
            img = Image.open(selfie.stream).convert("RGB")
            generated_dir = Path(app.root_path) / "static" / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            filename = "stylist_user.png"
            img.save(generated_dir / filename)
            skin  = detect_skin_tone_pil(img)
            style = generate_style_text_suggestions(
                image_pil=img, skin_label=skin.skin_label,
                dominant_hex=skin.dominant_hex,
                height_cm=height_cm, body_type=body_type, gender=gender)
            result = {
                "image_url": url_for("static", filename=f"generated/{filename}"),
                "skin": {"label": skin.skin_label,
                         "confidence": skin.confidence,
                         "dominant_hex": skin.dominant_hex},
                "palette": skin.best_palette, "style": style,
            }
            return render_template("stylist.html", result=result)
        except Exception as exc:
            flash(f"AI Stylist failed: {exc}", "error")
            return render_template("stylist.html", result=None)

    @app.route("/api/ai/style", methods=["POST"])
    def ai_style():
        try:
            recs = []
            for p in base_products[:5]:
                recs.append({"product": {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "image_src": url_for("static", filename=p.static_filename),
                }, "score": round(p.sustainability_score, 3)})
            return {"ok": True, "recs": recs}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Image Upload Try-On ───────────────────────────────────────────────────
    @app.route("/tryon/<int:product_id>", methods=["GET", "POST"])
    def tryon_product(product_id: int):
        product = base_by_id.get(product_id)
        if not product:
            flash("Product not found.", "error")
            return redirect(url_for("home"))
        cloth_path = Path(app.root_path) / "static" / product.static_filename
        if request.method == "GET":
            return render_tryon_page(product, result_image_url=None)
        if "person_image" not in request.files:
            flash("Please upload your photo.", "error")
            return render_tryon_page(product, result_image_url=None)
        if not cloth_path.exists():
            flash(f"Cloth image not found: {cloth_path}", "error")
            return render_tryon_page(product, result_image_url=None)
        person_file = request.files["person_image"]
        if not person_file.filename:
            flash("Invalid person image.", "error")
            return render_tryon_page(product, result_image_url=None)
        person_ext = _secure_ext(person_file.filename)
        with tempfile.TemporaryDirectory() as tmp_dir:
            person_path = Path(tmp_dir) / f"person{person_ext}"
            person_file.save(person_path)
            try:
                out_img = call_vton_space(
                    str(person_path), str(cloth_path), product.product_name)
            except Exception as exc:
                flash(f"Try-on failed: {exc}", "error")
                return render_tryon_page(product, result_image_url=None)
            generated_dir = Path(app.root_path) / "static" / "generated"
            generated_dir.mkdir(parents=True, exist_ok=True)
            out_file = generated_dir / f"tryon_{product_id}.png"
            out_img.save(out_file)
        return render_tryon_page(product,
            result_image_url=url_for("static",
                filename=f"generated/tryon_{product_id}.png"))

    # ── Live Camera Try-On page ───────────────────────────────────────────────
    @app.route("/live-tryon/<int:product_id>")
    def live_tryon_page(product_id: int):
        product = base_by_id.get(product_id)
        if not product:
            flash("Product not found.", "error")
            return redirect(url_for("home"))
        return render_template("live_tryon.html", product={
            "product_id":   product.product_id,
            "product_name": product.product_name,
            "image_src":    url_for("static", filename=product.static_filename),
        })

    # ── Live camera capture — uses IDM-VTON, fallback to OpenCV ──────────────
    @app.route("/live-tryon/capture-form/<int:product_id>", methods=["POST"])
    def live_tryon_capture_form(product_id: int):
        product = base_by_id.get(product_id)
        if not product:
            return jsonify({"ok": False, "error": "Product not found"})

        person_file = request.files.get("person_image")
        if not person_file or not person_file.filename:
            return jsonify({"ok": False, "error": "No image received"})

        cloth_path = Path(app.root_path) / "static" / product.static_filename
        if not cloth_path.exists():
            return jsonify({"ok": False, "error": f"Cloth image not found: {cloth_path}"})

        generated_dir = Path(app.root_path) / "static" / "generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        uid      = uuid.uuid4().hex[:8]
        out_path = str(generated_dir / f"live_result_{uid}.jpg")

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                person_path = str(Path(tmp_dir) / "capture.jpg")
                person_file.save(person_path)

                # ── Try IDM-VTON first ────────────────────────────────────
                try:
                    out_img = call_vton_space(
                        person_path,
                        str(cloth_path),
                        product.product_name,
                    )
                    out_img.convert("RGB").save(out_path)
                    print("[LiveTryon] IDM-VTON success!")

                except Exception as vton_err:
                    # ── Fallback to OpenCV overlay ────────────────────────
                    print(f"[LiveTryon] IDM-VTON failed: {vton_err}")
                    print("[LiveTryon] Using OpenCV fallback...")
                    success = apply_cloth_to_photo(
                        person_path,
                        str(cloth_path),
                        out_path,
                    )
                    if not success:
                        return jsonify({
                            "ok": False,
                            "error": "Try-on failed. Please stand back so your full body is visible."
                        })
                    print("[LiveTryon] OpenCV fallback success!")

            with open(out_path, "rb") as f:
                result_b64 = base64.b64encode(f.read()).decode()

            return jsonify({
                "ok":     True,
                "result": f"data:image/jpeg;base64,{result_b64}",
            })

        except Exception as e:
            print(f"[LiveTryon] Unexpected error: {e}")
            return jsonify({"ok": False, "error": str(e)})

    # ── Size Recommendation ───────────────────────────────────────────────────
    @app.route("/size", methods=["GET", "POST"])
    def size_page():
        if request.method == "GET":
            return render_template("size.html", result=None, error=None)
        try:
            height_ft  = float(request.form.get("height_ft", 0))
            height_in  = float(request.form.get("height_in", 0))
            weight     = float(request.form.get("weight", 0))
            gender_str = request.form.get("gender", "Female")
            gender     = 0 if gender_str == "Female" else 1
        except (TypeError, ValueError):
            return render_template("size.html", result=None,
                error="Please enter valid height and weight.")
        height_cm = round((height_ft * 12 + height_in) * 2.54, 1)
        if not (100 <= height_cm <= 230):
            return render_template("size.html", result=None,
                error="Please enter a valid height (3ft 3in – 7ft 6in).")
        if not (20 <= weight <= 250):
            return render_template("size.html", result=None,
                error="Please enter a valid weight (20–250 kg).")
        image_file = request.files.get("front_image")
        if not image_file or not image_file.filename:
            return render_template("size.html", result=None,
                error="Please upload a front-facing photo.")
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return render_template("size.html", result=None,
                error="Could not read image. Please try another photo.")
        measurements = get_body_measurements(img, height_cm)
        if not measurements:
            return render_template("size.html", result=None,
                error="Body not detected. Use a clear full-body front photo.")
        try:
            size = recommend_size(measurements, height_cm, weight, gender)
        except Exception as e:
            return render_template("size.html", result=None,
                                   error=f"Size prediction failed: {e}")
        result = {
            "size": size, "height_cm": height_cm,
            "weight": weight, "gender": gender_str,
            "shoulder":    measurements["shoulder"],
            "chest":       measurements["chest"],
            "waist":       measurements["waist"],
            "hip":         measurements["hip"],
            "chest_depth": measurements["chest_depth"],
            "hip_depth":   measurements["hip_depth"],
        }
        return render_template("size.html", result=result, error=None)

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)