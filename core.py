import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastai.vision.all import PILImage as FAI_PILImage
import pydicom
import matplotlib.cm as cm_plt

def load_dicom(file_bytes):
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array.astype(np.float32)
    if arr.max() > arr.min():
        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    else:
        arr = np.zeros_like(arr, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr)

def check_image_quality(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    r, g, b = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
    is_grey = max(abs(r-g), abs(g-b), abs(r-b)) < 30
    w, h = pil_img.size
    aspect_ok = 0.5 < (w/h) < 2.0
    size_ok = w >= 100 and h >= 100
    return is_grey and aspect_ok and size_ok

def compute_gradcam(learner, pil_img):
    model = learner.model.eval()
    grads, acts = [], []
    layer = model[0][-1]
    hf = layer.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    hb = layer.register_full_backward_hook(lambda m, i, o: grads.append(o[0].detach()))
    try:
        fai_img = FAI_PILImage.create(np.array(pil_img))
        dl = learner.dls.test_dl([fai_img], num_workers=0)
        xb, _ = next(iter(dl))
        logits = model(xb)
        model.zero_grad()
        pred_class = logits.argmax(dim=1).item()
        logits[0, pred_class].backward()
        if not grads:
            return np.zeros((224, 224), dtype=np.float32)
        weights = grads[0].mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * acts[0].squeeze(0)).sum(0)).cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        return np.array(cam_img) / 255.0
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return np.zeros((224, 224), dtype=np.float32)
    finally:
        hf.remove()
        hb.remove()

def overlay_heatmap(pil_img, cam, alpha=0.45):
    base = pil_img.convert("RGB").resize((224, 224))
    heatmap = Image.fromarray((cm_plt.jet(cam)[:, :, :3] * 255).astype(np.uint8))
    return Image.blend(base, heatmap, alpha)

def run_tta(learner, pil_img, n=10):
    probs = []
    for _ in range(n):
        img_np = np.array(pil_img).astype(np.float32)
        img_np += np.random.uniform(-15, 15)
        factor = np.random.uniform(0.85, 1.15)
        mean = img_np.mean()
        img_np = (img_np - mean) * factor + mean
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        aug_img = FAI_PILImage.create(img_np)
        with learner.no_bar(), learner.no_logging():
            _, _, p = learner.predict(aug_img)
        probs.append(float(p[1]))
    a = np.array(probs)
    return dict(
        mean=float(a.mean()), 
        std=float(a.std()),
        votes_pneu=int((a > .5).sum()),
        votes_norm=int((a <= .5).sum())
    )