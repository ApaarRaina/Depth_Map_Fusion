import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

warnings.filterwarnings("ignore")
DATASET_ROOT   = "./dataset"
BATCH_SIZE     = 4
NUM_WORKERS    = 2
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CMAP_INDOOR    = "inferno"
CMAP_OUTDOOR   = "plasma"

print(f"Using device: {DEVICE}")


class DepthDataset(Dataset):
    IMG_H, IMG_W = 384, 512

    def __init__(self, split: str):
        assert split in ("train", "test")
        self.img_dir   = Path(DATASET_ROOT) / split / "images"
        self.depth_dir = Path(DATASET_ROOT) / split / "depth"

        self.samples = sorted([
            p for p in self.img_dir.glob("*.png")
            if (self.depth_dir / p.name).exists()
        ])

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found in {self.img_dir}. "
                "Please run the download script first."
            )

        self.to_tensor = transforms.Compose([
            transforms.Resize((self.IMG_H, self.IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        print(f"  [{split}] {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path   = self.samples[idx]
        name   = path.stem
        source = "indoor" if name.startswith("indoor") else "outdoor"

        # RGB
        rgb_pil = Image.open(path).convert("RGB")
        rgb_raw = np.array(rgb_pil.resize((self.IMG_W, self.IMG_H)))
        rgb_t   = self.to_tensor(rgb_pil)

        depth_raw = np.array(
            Image.open(self.depth_dir / path.name), dtype=np.float32
        )
        scale    = 1000.0 if source == "indoor" else 256.0
        depth_gt = (depth_raw / scale).astype(np.float32)

        return {
            "rgb_tensor": rgb_t,
            "rgb_raw":    rgb_raw,
            "depth_gt":   depth_gt,
            "source":     source,
            "name":       name,
        }

def collate_fn(batch):
    return {
        "rgb_tensor": torch.stack([b["rgb_tensor"] for b in batch]),
        "rgb_raw":    [b["rgb_raw"]   for b in batch],
        "depth_gt":   [b["depth_gt"]  for b in batch],
        "source":     [b["source"]    for b in batch],
        "name":       [b["name"]      for b in batch],
    }

def build_loaders():
    print("\n── Building DataLoaders ──")
    train_ds = DepthDataset("train")
    test_ds  = DepthDataset("test")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda"),
    )

    print(f"  Train loader: {len(train_loader)} batches × {BATCH_SIZE}")
    print(f"  Test  loader: {len(test_loader)}  batches × {BATCH_SIZE}")
    return train_loader, test_loader



class MiDaSIndoor:
    NAME = "MiDaS (DPT_Large) — Indoor"

    def __init__(self, device):
        print("\n── Loading MiDaS DPT_Large (indoor model) ──")
        self.device = device
        self.model  = torch.hub.load(
            "intel-isl/MiDaS", "DPT_Large", pretrained=True
        ).to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform   = midas_transforms.dpt_transform
        print("MiDaS ready.")

    @torch.no_grad()
    def predict(self, rgb_list: list[np.ndarray]) -> list[np.ndarray]:
        results = []
        for rgb in rgb_list:
            inp    = self.transform(rgb).to(self.device)
            pred   = self.model(inp)
            pred   = pred.squeeze().cpu().numpy().astype(np.float32)
            pred   = 1.0 / (pred + 1e-6)
            pred   = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            results.append(pred)
        return results


class DepthAnythingOutdoor:
    NAME = "Depth‑Anything V2 Large — Outdoor (HF)"

    def __init__(self, device):
        print("\n── Loading Depth‑Anything V2 Large (outdoor model) ──")
        self.device = device

        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model     = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        print("Depth‑Anything Outdoor ready.")

    @torch.no_grad()
    def predict(self, rgb_list: list[np.ndarray]) -> list[np.ndarray]:
        results = []
        for rgb in rgb_list:
            pil = Image.fromarray(rgb)
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device)

            outputs = self.model(**inputs)
            depth = outputs.predicted_depth

            # upsample to match original
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(pil.height, pil.width),
                mode="bicubic",
                align_corners=False
            ).squeeze()

            pred = depth.cpu().numpy().astype(np.float32)
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            results.append(pred)
        return results

# -------------------------------
# Utilities
# -------------------------------
def colorize(depth: np.ndarray, cmap_name: str) -> np.ndarray:
    d = depth.copy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cmap   = plt.get_cmap(cmap_name)
    colored = (cmap(d)[:, :, :3] * 255).astype(np.uint8)
    return colored

def display_batch(batch, midas_preds, outdoor_preds):
    n   = len(batch["name"])
    fig = plt.figure(figsize=(20, 5 * n), facecolor="#0d0d0d")
    fig.suptitle(
        "Depth Estimation — Mixed Indoor / Outdoor Batch",
        fontsize=18, color="white", fontweight="bold", y=1.01
    )

    gs  = gridspec.GridSpec(n, 4, figure=fig,
                            wspace=0.03, hspace=0.25)

    col_titles = ["RGB Input", "MiDaS (Indoor)", "Outdoor Model", "Ground Truth"]

    for row in range(n):
        source  = batch["source"][row]
        name    = batch["name"][row]
        rgb     = batch["rgb_raw"][row]
        gt      = batch["depth_gt"][row]
        midas_d = midas_preds[row]
        outdoor_d = outdoor_preds[row]

        cmap = CMAP_INDOOR if source == "indoor" else CMAP_OUTDOOR

        panels = [
            rgb,
            colorize(midas_d, CMAP_INDOOR),
            colorize(outdoor_d, CMAP_OUTDOOR),
            colorize(gt, cmap),
        ]

        for col, img in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=11, fontweight="bold", pad=6)

            if col == 0:
                label = f"[{source.upper()}]\n{name}"
                ax.set_ylabel(label, color="#aaaaaa", fontsize=8,
                              rotation=0, labelpad=70, va="center")

            border_color = "#4fc3f7" if source == "indoor" else "#a5d6a7"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(1.5)

    legend_text = (
        "Blue border = Indoor (NYU)  |  "
        "Green border = Outdoor (KITTI)  |  "
        f"Batch size = {n}  |  Device = {DEVICE.upper()}"
    )
    fig.text(0.5, -0.01, legend_text, ha="center",
             color="#888888", fontsize=9)

    plt.tight_layout()
    plt.savefig("depth_output.png", dpi=120, bbox_inches="tight",
                facecolor="#0d0d0d")
    print("\nSaved visualisation → depth_output.png")
    plt.show()

def run_inference_and_display(loader, midas: MiDaSIndoor, outdoor: DepthAnythingOutdoor, split_name: str = "test"):
    print(f"\n── Running inference on one {split_name} batch ──")

    batch = next(iter(loader))
    rgb_list = batch["rgb_raw"]

    print(f"  Batch sources: {batch['source']}")
    print(f"  Running MiDaS  ...")
    midas_preds = midas.predict(rgb_list)

    print(f"  Running Outdoor model ...")
    outdoor_preds   = outdoor.predict(rgb_list)

    display_batch(batch, midas_preds, outdoor_preds)

# -------------------------------
# Main
# -------------------------------
def main():
    print("=" * 60)
    print("  Depth Estimation Pipeline")
    print("  NYU (indoor) + KITTI (outdoor) mixed batch inference")
    print("=" * 60)

    train_loader, test_loader = build_loaders()

    midas = MiDaSIndoor(DEVICE)
    outdoor = DepthAnythingOutdoor(DEVICE)

    run_inference_and_display(test_loader, midas, outdoor, split_name="test")

if __name__ == "__main__":
    main()
