import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.hub
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Models
# =========================

class MiDaS:
    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True).to(DEVICE).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    @torch.no_grad()
    def predict(self, rgb):
        inp = self.transform(rgb).to(DEVICE)
        pred = self.model(inp).squeeze().cpu()

        pred = 1.0 / (pred + 1e-6)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return pred


class DepthAnything:
    def __init__(self):
        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE).eval()

    @torch.no_grad()
    def predict(self, rgb):
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(DEVICE)

        depth = self.model(**inputs).predicted_depth

        depth = F.interpolate(
            depth.unsqueeze(1),
            size=(pil.height, pil.width),
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth.cpu()


# =========================
# Saving utilities
# =========================

def save_depth_16bit(depth_tensor, path):
    depth = depth_tensor.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint16 = (depth * 65535).astype(np.uint16)
    cv2.imwrite(str(path), depth_uint16)


def save_depth_viz(depth_tensor, path):
    depth = depth_tensor.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(path), colored)


# =========================
# Processing
# =========================

def process_split(split, input_root, output_root, midas, da_model):
    img_dir = Path(input_root) / split / "images"
    gt_dir  = Path(input_root) / split / "depth"

    out_rgb = Path(output_root) / split / "rgb"
    out_d1  = Path(output_root) / split / "depth_1"
    out_d2  = Path(output_root) / split / "depth_2"
    out_gt  = Path(output_root) / split / "depth_gt"
    out_viz = Path(output_root) / split / "viz"

    for d in [out_rgb, out_d1, out_d2, out_gt, out_viz]:
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(img_dir.glob("*.png"))

    for path in tqdm(files, desc=f"Processing {split}"):

        rgb_pil = Image.open(path).convert("RGB")
        rgb_np  = np.array(rgb_pil)

        d1 = midas.predict(rgb_np)
        d2 = da_model.predict(rgb_np)

        H, W = 256, 256

        rgb_resized = rgb_pil.resize((W, H))

        d1 = F.interpolate(d1.unsqueeze(0).unsqueeze(0),
                           size=(H, W),
                           mode="bilinear",
                           align_corners=False).squeeze()

        d2 = F.interpolate(d2.unsqueeze(0).unsqueeze(0),
                           size=(H, W),
                           mode="bilinear",
                           align_corners=False).squeeze()

        gt_path = gt_dir / path.name
        if not gt_path.exists():
            continue

        gt = np.array(Image.open(gt_path), dtype=np.float32)

        source = "indoor" if path.stem.startswith("indoor") else "outdoor"
        scale = 1000.0 if source == "indoor" else 256.0
        gt = torch.from_numpy(gt / scale)

        gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0),
                           size=(H, W),
                           mode="nearest").squeeze()

        name = path.stem

        rgb_resized.save(out_rgb / f"{name}.png")

        save_depth_16bit(d1, out_d1 / f"{name}.png")
        save_depth_16bit(d2, out_d2 / f"{name}.png")
        save_depth_16bit(gt, out_gt / f"{name}.png")

        save_depth_viz(d1, out_viz / f"{name}_d1.png")
        save_depth_viz(d2, out_viz / f"{name}_d2.png")
        save_depth_viz(gt, out_viz / f"{name}_gt.png")


# =========================
# Main
# =========================

def main():
    input_root  = "./dataset"
    output_root = "./processed_dataset"

    midas = MiDaS()
    da_model = DepthAnything()

    process_split("train", input_root, output_root, midas, da_model)
    process_split("test", input_root, output_root, midas, da_model)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
