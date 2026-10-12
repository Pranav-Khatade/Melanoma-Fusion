# Multimodal melanoma classifier (image + clinical metadata)

Implements the architecture from the draft: EfficientNet (GAP) ‖ metadata MLP → concat → Dense/BN/ReLU/Dropout → sigmoid.

## Setup
    pip install -r requirements.txt

## Data
Download SIIM-ISIC 2020 (JPEG + train.csv) from Kaggle / ISIC archive. Columns used:
`image_name, patient_id, sex, age_approx, anatom_site_general_challenge, target`.

## Train (fusion) and baseline (image-only) with identical splits/seed
    python train.py --csv train.csv --img_dir jpeg/train --out runs/fusion
    python train.py --csv train.csv --img_dir jpeg/train --out runs/imgonly --image_only
Add `--target_sens 0.95` to pick a screening threshold with ≥95% validation sensitivity.
Results (val/test AUC, sens, spec, F1) go to `runs/*/results.json`.

## Serve
    RUN_DIR=runs/fusion uvicorn app:app --port 8000
    curl -F image=@lesion.jpg -F age=55 -F sex=male -F site="torso" localhost:8000/predict

## Explainability
`explain.py`: `grad_cam(...)` heatmaps, `metadata_shap(...)` per-feature SHAP for age/sex/site.

## Design choices vs. the draft
- Patient-level 70/15/15 split (no patient in two splits) – prevents leakage.
- Class imbalance via weighted sampler instead of balancing the *test* set; evaluate on the natural prevalence.
- Threshold chosen on validation only.
- Metadata dropout during training so the model degrades gracefully when metadata is missing.
- Not a medical device; clinical use needs prospective validation and regulatory clearance.
