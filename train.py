import argparse, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import (MelanomaDataset, build_meta_preprocessor, get_transforms,
                  patient_level_split, save_preprocessor)
from model import FusionNet


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=(tp + tn) / len(y), sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                prec=tp / max(tp + fp, 1), f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, p), thr=float(thr))


def pick_threshold(y, p, target_sens=None):
    grid = np.linspace(0.05, 0.95, 91)
    if target_sens:  # highest threshold that still reaches the sensitivity target
        ok = [t for t in grid if metrics_at(y, p, t)["sens"] >= target_sens]
        if ok:
            return max(ok)
    j = [metrics_at(y, p, t)["sens"] + metrics_at(y, p, t)["spec"] - 1 for t in grid]  # Youden J
    return grid[int(np.argmax(j))]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, m, y in loader:
        ps.append(torch.sigmoid(model(x.to(device), m.to(device))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True); a.add_argument("--img_dir", required=True)
    a.add_argument("--img_col", default="image_name"); a.add_argument("--label_col", default="target")
    a.add_argument("--group_col", default="patient_id"); a.add_argument("--ext", default=".jpg")
    a.add_argument("--age_col", default="age_approx"); a.add_argument("--sex_col", default="sex")
    a.add_argument("--site_col", default="anatom_site_general_challenge")
    a.add_argument("--backbone", default="efficientnet_b0"); a.add_argument("--size", type=int, default=224)
    a.add_argument("--epochs", type=int, default=30); a.add_argument("--bs", type=int, default=32)
    a.add_argument("--lr", type=float, default=3e-4); a.add_argument("--wd", type=float, default=1e-2)
    a.add_argument("--patience", type=int, default=6); a.add_argument("--target_sens", type=float, default=None)
    a.add_argument("--image_only", action="store_true"); a.add_argument("--workers", type=int, default=4)
    a.add_argument("--out", default="runs/fusion"); a.add_argument("--seed", type=int, default=42)
    a = a.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(a.csv)
    tr, va, te = patient_level_split(df, a.group_col, a.label_col, seed=a.seed)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | pos rate {tr[a.label_col].mean():.4f}")

    prep = build_meta_preprocessor(a.age_col, a.sex_col, a.site_col).fit(tr)  # fit on train only
    save_preprocessor(prep, f"{a.out}/meta_prep.joblib")
    M = lambda d: prep.transform(d) if not a.image_only else np.zeros((len(d), 0))
    meta_dim = M(tr.head(2)).shape[1]

    mk = lambda d, train: MelanomaDataset(d, a.img_dir, a.img_col, a.label_col, M(d),
                                          get_transforms(a.size, train), a.size, a.ext)
    ds_tr, ds_va, ds_te = mk(tr, True), mk(va, False), mk(te, False)

    # Handle class imbalance with a weighted sampler + pos_weight (no synthetic SMOTE images needed)
    y_tr = ds_tr.labels
    w = np.where(y_tr == 1, 0.5 / y_tr.mean(), 0.5 / (1 - y_tr.mean()))
    sampler = WeightedRandomSampler(w, len(w), replacement=True)
    dl = lambda ds, **k: DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, **k)
    L_tr, L_va, L_te = dl(ds_tr, sampler=sampler), dl(ds_va), dl(ds_te)

    model = FusionNet(meta_dim, a.backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    best_auc, bad = -1, 0
    for ep in range(a.epochs):
        model.train(); run = 0
        for x, m, y in L_tr:
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = crit(model(x, m), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * len(y)
        sched.step()
        yv, pv = predict(model, L_va, dev)
        auc = roc_auc_score(yv, pv)
        print(f"ep {ep+1:02d} loss {run/len(ds_tr):.4f} val_auc {auc:.4f}")
        if auc > best_auc:
            best_auc, bad = auc, 0
            torch.save({"state": model.state_dict(), "meta_dim": meta_dim, "backbone": a.backbone,
                        "size": a.size, "image_only": a.image_only}, f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping"); break

    ck = torch.load(f"{a.out}/best.pt", map_location=dev); model.load_state_dict(ck["state"])
    yv, pv = predict(model, L_va, dev)
    thr = pick_threshold(yv, pv, a.target_sens)       # threshold chosen on VAL, never on test
    yt, pt = predict(model, L_te, dev)
    res = {"val": metrics_at(yv, pv, thr), "test": metrics_at(yt, pt, thr)}
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    ck["threshold"] = float(thr); torch.save(ck, f"{a.out}/best.pt")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
