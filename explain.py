import cv2
import numpy as np
import torch


def grad_cam(model, img_t, meta_t, device="cpu"):
    model.eval()
    img_t, meta_t = img_t.to(device), meta_t.to(device)
    with torch.enable_grad():
        logit, fmap = model(img_t, meta_t, return_fmap=True)
        fmap.retain_grad()
        logit.sum().backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fmap).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=img_t.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay(rgb_uint8, cam, alpha=0.4):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb_uint8, (cam.shape[1], cam.shape[0]))
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def metadata_shap(model, img_t, meta_t, background_meta, feature_names, device="cpu", nsamples=200):
    import shap
    model.eval()
    img_t = img_t.to(device)

    def f(m):
        m = torch.tensor(m, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(img_t.expand(len(m), -1, -1, -1), m)).cpu().numpy()

    ex = shap.KernelExplainer(f, background_meta)
    sv = ex.shap_values(meta_t.cpu().numpy(), nsamples=nsamples)
    return dict(zip(feature_names, np.array(sv).reshape(-1).tolist()))
