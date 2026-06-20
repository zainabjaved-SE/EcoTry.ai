# File: ai_stylist.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
except Exception:
    cv2 = None
    np = None
    mp = None
    KMeans = None


@dataclass(frozen=True)
class SkinToneResult:
    skin_label: str
    confidence: int
    dominant_hex: str
    dominant_lab: Tuple[float, float, float]
    best_palette: Dict[str, Any]


def _palette_db() -> List[Dict[str, Any]]:
    return [
        {
            "id": "warm_neutrals_olive",
            "name": "Warm Neutrals + Olive",
            "tags": ["warm", "everyday", "soft-contrast"],
            "colors_hex": ["#F4E3D7", "#D7B49E", "#A97C50", "#6B7B3E", "#2F2B28"],
            "mean_lab": (190.0, 140.0, 150.0),
        },
        {
            "id": "cool_minimal",
            "name": "Cool Minimal",
            "tags": ["cool", "minimal", "clean"],
            "colors_hex": ["#F6F7FB", "#DDE2EA", "#97A1B2", "#2F3A4C", "#0E0F12"],
            "mean_lab": (205.0, 128.0, 125.0),
        },
        {
            "id": "bold_jewel",
            "name": "Bold Jewel Tones",
            "tags": ["bold", "evening", "high-contrast"],
            "colors_hex": ["#0B3D91", "#0E7C7B", "#7D1538", "#3B1F2B", "#F2E9E4"],
            "mean_lab": (165.0, 135.0, 140.0),
        },
        {
            "id": "earthy_rich",
            "name": "Earthy Rich",
            "tags": ["warm", "earthy", "autumn"],
            "colors_hex": ["#7A4E2D", "#C57B57", "#F1AB86", "#2D3A2E", "#E7D7C1"],
            "mean_lab": (175.0, 145.0, 155.0),
        },
        {
            "id": "deep_contrast",
            "name": "Deep Contrast",
            "tags": ["cool", "deep", "contrast"],
            "colors_hex": ["#111827", "#1F2937", "#0EA5E9", "#DC2626", "#F9FAFB"],
            "mean_lab": (150.0, 130.0, 120.0),
        },
    ]


def _lab_to_hex(lab: "np.ndarray") -> str:
    lab_1x1 = lab.reshape(1, 1, 3).astype("uint8")
    bgr = cv2.cvtColor(lab_1x1, cv2.COLOR_LAB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(3)
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _skin_label_from_lab(dominant_lab: "np.ndarray") -> str:
    L, a, b = float(dominant_lab[0]), float(dominant_lab[1]), float(dominant_lab[2])
    tone = "Light" if L > 185 else "Medium" if L > 140 else "Deep"
    undertone = "Warm" if b > a else "Cool"
    return f"{undertone} / {tone}"


def _recommend_best_palette(dominant_lab: Tuple[float, float, float]) -> Dict[str, Any]:
    db = _palette_db()
    dl = np.array(dominant_lab, dtype=np.float32)

    scored = []
    for p in db:
        pl = np.array(p["mean_lab"], dtype=np.float32)
        d = float(np.linalg.norm(pl - dl))
        score = float(np.exp(-d / 35.0))
        scored.append((score, p))

    scored.sort(key=lambda x: -x[0])
    best_score, best = scored[0]
    return {
        "id": best["id"],
        "name": best["name"],
        "score": best_score,
        "tags": best["tags"],
        "colors_hex": best["colors_hex"],
    }


def _expand_bbox(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, scale: float = 1.5
) -> Tuple[int, int, int, int]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = int(max(0, cx - bw / 2))
    ny1 = int(max(0, cy - bh / 2))
    nx2 = int(min(w - 1, cx + bw / 2))
    ny2 = int(min(h - 1, cy + bh / 2))
    return nx1, ny1, nx2, ny2


def _detect_face_crop(rgb: "np.ndarray") -> Optional["np.ndarray"]:
    h, w, _ = rgb.shape
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.35
    )
    res = detector.process(rgb)
    if not res.detections:
        return None

    det = res.detections[0]
    bbox = det.location_data.relative_bounding_box
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, w, h, scale=1.50)
    crop = rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return crop


def _cheek_mask(face_rgb: "np.ndarray", face_landmarks: Any) -> "np.ndarray":
    h, w, _ = face_rgb.shape
    left_ids = [234, 93, 132, 58, 172]
    right_ids = [454, 323, 361, 288, 397]

    def pts(ids: List[int]) -> "np.ndarray":
        out = []
        for i in ids:
            lm = face_landmarks.landmark[i]
            out.append([int(lm.x * w), int(lm.y * h)])
        return np.array(out, dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts(left_ids), 255)
    cv2.fillConvexPoly(mask, pts(right_ids), 255)
    return mask


def detect_skin_tone_pil(selfie_pil: Image.Image) -> SkinToneResult:
    if cv2 is None or np is None or KMeans is None or mp is None:
        raise RuntimeError(
            "Missing deps. Install: pip install numpy opencv-python mediapipe scikit-learn"
        )

    rgb = np.array(selfie_pil.convert("RGB"))
    rgb = np.ascontiguousarray(rgb)

    face_rgb = _detect_face_crop(rgb)
    if face_rgb is None:
        raise ValueError(
            "Face not detected. Use a clearer front-facing selfie in good light."
        )

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
    )
    res = mesh.process(face_rgb)
    if not res.multi_face_landmarks:
        raise ValueError("Face landmarks not detected. Try a clearer selfie.")

    face = res.multi_face_landmarks[0]
    mask = _cheek_mask(face_rgb, face)

    bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    skin_pixels = lab[mask == 255].reshape(-1, 3)
    if skin_pixels.shape[0] < 180:
        raise ValueError("Not enough cheek pixels. Try brighter lighting.")

    k = 3 if skin_pixels.shape[0] > 1500 else 2
    km = KMeans(n_clusters=k, n_init=6, random_state=42)
    labels = km.fit_predict(skin_pixels)
    counts = np.bincount(labels)
    dominant_lab = km.cluster_centers_[int(np.argmax(counts))].astype(np.float32)

    dominant_hex = _lab_to_hex(dominant_lab)
    label = _skin_label_from_lab(dominant_lab)
    confidence = int(min(100, 55 + (skin_pixels.shape[0] / 1500.0) * 45))

    best_palette = _recommend_best_palette(
        (float(dominant_lab[0]), float(dominant_lab[1]), float(dominant_lab[2]))
    )

    return SkinToneResult(
        skin_label=label,
        confidence=confidence,
        dominant_hex=dominant_hex,
        dominant_lab=(
            float(dominant_lab[0]),
            float(dominant_lab[1]),
            float(dominant_lab[2]),
        ),
        best_palette=best_palette,
    )


def _pose_insights(image_pil: Image.Image) -> Dict[str, Any]:
    if cv2 is None or np is None or mp is None:
        return {"pose_detected": False}

    bgr = cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return {"pose_detected": False}

    lm = res.pose_landmarks.landmark
    L_SH, R_SH = lm[11], lm[12]
    L_HIP, R_HIP = lm[23], lm[24]

    def dist(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    shoulder_w = dist(L_SH, R_SH)
    hip_w = dist(L_HIP, R_HIP)
    ratio = shoulder_w / (hip_w + 1e-6)

    if ratio > 1.08:
        shape = "Inverted Triangle (shoulders broader)"
    elif ratio < 0.92:
        shape = "Pear (hips broader)"
    else:
        shape = "Balanced"

    return {"pose_detected": True, "shoulder_to_hip_ratio": float(ratio), "shape": shape}


def generate_style_text_suggestions(
    *,
    image_pil: Image.Image,
    skin_label: str,
    dominant_hex: str,
    height_cm: Optional[float],
    body_type: str,
    gender: str,
) -> Dict[str, List[str]]:
    pose = _pose_insights(image_pil)

    recs: List[str] = []
    avoid: List[str] = []

    recs.append("Prefer solid colors + 1 accent piece (premium, clean look).")
    recs.append("Use vertical lines (open jacket, long cardigan, straight seams) to look taller.")

    if height_cm is not None:
        if height_cm < 160:
            recs.append("Short height: high-waist jeans + shorter jackets elongate legs.")
            avoid.append("Avoid long oversized tops that cut the leg line.")
        elif height_cm > 175:
            recs.append("Tall height: layering (overshirt, long coat) looks balanced.")
            avoid.append("Avoid ultra-short cropped tops if you want a formal silhouette.")

    bt = (body_type or "Average").lower()
    if bt == "slim":
        recs.append("Slim build: add structure with overshirts/blazers and light layering.")
        recs.append("Bottoms: straight or relaxed jeans work best.")
    elif bt == "athletic":
        recs.append("Athletic build: semi-fitted tops + straight jeans/trousers.")
        avoid.append("Avoid extremely oversized outfits that hide proportions.")
    elif bt == "heavy":
        recs.append("Heavier build: darker solids + vertical patterns + structured outerwear.")
        recs.append("Bottoms: straight-leg or tapered jeans are flattering.")
        avoid.append("Avoid loud horizontal stripes across the midsection.")

    if pose.get("pose_detected"):
        ratio = float(pose["shoulder_to_hip_ratio"])
        shape = pose["shape"]
        recs.append(f"Body proportion: {shape} (ratio {ratio:.2f}).")

        if ratio > 1.08:
            recs.append("Patterns: vertical stripes recommended; avoid heavy shoulder details.")
            recs.append("Jeans: straight/wide-leg to balance shoulders.")
            recs.append("Shirt length: hip-length tops balance the frame.")
            avoid.append("Avoid tight crew-necks and shoulder pads.")
        elif ratio < 0.92:
            recs.append("Patterns: texture/stripes on top are okay; keep bottoms simpler/darker.")
            recs.append("Jeans: straight jeans; avoid ultra-skinny if you want balance.")
            recs.append("Tops: structured jackets help balance hips.")
            avoid.append("Avoid heavy prints on bottoms if you want balance.")
        else:
            recs.append("Balanced: you can wear both fitted and relaxed silhouettes confidently.")
            recs.append("Try: straight jeans + either cropped or hip-length tops.")
    else:
        recs.append("Tip: upload a full-body photo for more accurate pattern/fit advice.")

    if "Warm" in (skin_label or ""):
        recs.append("Warm undertone: olive, cream, tan, warm browns, mustard accents look best.")
        avoid.append("Avoid very icy blues if you want a warm glow.")
    elif "Cool" in (skin_label or ""):
        recs.append("Cool undertone: navy, charcoal, crisp white, emerald, cobalt accents look best.")
        avoid.append("Avoid very yellowish oranges if they dull the skin.")

    if gender.lower() == "female":
        recs.append("Outfit idea: high-waist jeans + structured top + minimal accessories.")
    elif gender.lower() == "male":
        recs.append("Outfit idea: straight jeans + overshirt/blazer + clean sneakers.")

    return {"recommendations": recs, "avoid": avoid}