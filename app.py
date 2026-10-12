import base64, io, os
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from data import get_transforms, load_preprocessor, preprocess_image
from explain import grad_cam, overlay
from model import FusionNet

RUN = os.environ.get("RUN_DIR", "runs/fusion")
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{RUN}/best.pt", map_location=dev)
model = FusionNet(ck["meta_dim"], ck["backbone"], pretrained=False).to(dev)
model.load_state_dict(ck["state"]); model.eval()
prep, thr, size = load_preprocessor(f"{RUN}/meta_prep.joblib"), ck.get("threshold", 0.5), ck["size"]
tf = get_transforms(size, train=False)

app = FastAPI(title="Melanoma multimodal triage (research prototype)")


@app.post("/predict")
async def predict(image: UploadFile = File(...), age: float | None = Form(None),
                  sex: str | None = Form(None), site: str | None = Form(None)):
    pil = preprocess_image(Image.open(io.BytesIO(await image.read())), size)
    if ck["image_only"]:
        meta = np.zeros((1, 0), dtype=np.float32)
    else:
        row = pd.DataFrame([{"age_approx": age, "sex": sex, "anatom_site_general_challenge": site}])
        meta = prep.transform(row).astype(np.float32)
    x, m = tf(pil).unsqueeze(0), torch.from_numpy(meta)
    cam = grad_cam(model, x, m, dev)
    with torch.no_grad():
        prob = torch.sigmoid(model(x.to(dev), m.to(dev))).item()
    buf = io.BytesIO(); Image.fromarray(overlay(np.array(pil), cam)).save(buf, format="PNG")
    return {"malignancy_probability": round(prob, 4), "flagged_for_review": prob >= thr,
            "threshold": thr, "gradcam_png_base64": base64.b64encode(buf.getvalue()).decode(),
            "disclaimer": "Research prototype. Not a medical device; not for clinical use."}
