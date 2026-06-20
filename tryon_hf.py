import os
from gradio_client import Client, file

VTON_SPACE_ID = "EcoTry/IDM-VTON"

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ✅ single clean client
client = Client(VTON_SPACE_ID, hf_token=HF_TOKEN)

def run_tryon(person_img_path, cloth_img_path, prompt="t-shirt"):
    result = client.predict(
        dict={
            "background": file(person_img_path),
            "layers": [],
            "composite": None
        },
        garm_img=file(cloth_img_path),
        garment_des=f"high quality photo of person wearing {prompt}",
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=30,
        seed=42,
        api_name="/tryon"
    )

    return result[0]