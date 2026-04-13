import os
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from GAN_Model import (
    ConfidenceGenerator,
    PatchDiscriminator,
    gradient_smoothness_loss,
    confidence_sum_loss,
    count_params,
    EPSILON
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET (UPDATED)
# =========================

class FusionDataset(Dataset):
    def __init__(self, split: str, data_root: str, img_size: int):
        self.root = Path(data_root) / split

        self.rgb_dir = self.root / "rgb"
        self.d1_dir  = self.root / "depth_1"
        self.d2_dir  = self.root / "depth_2"
        self.gt_dir  = self.root / "depth_gt"

        self.img_size = (img_size, img_size)

        self.samples = sorted(list(self.rgb_dir.glob("*.png")))
        if not self.samples:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")

        self.rgb_tf = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def load_depth(self, path):
        depth = np.array(Image.open(path)).astype(np.float32)

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return torch.from_numpy(depth).unsqueeze(0)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        name = path.stem

        rgb_pil = Image.open(path).convert("RGB")
        rgb = self.rgb_tf(rgb_pil)

        d1 = self.load_depth(self.d1_dir / f"{name}.png")
        d2 = self.load_depth(self.d2_dir / f"{name}.png")
        gt = self.load_depth(self.gt_dir / f"{name}.png")

        return {
            "rgb": rgb,
            "depth_1": d1,
            "depth_2": d2,
            "depth_gt": gt,
            "name": name
        }


# =========================
# LOADERS
# =========================

def build_loaders(args):
    print("\n── DataLoaders ──")

    train_ds = FusionDataset("train", args.data, args.img_size)
    test_ds  = FusionDataset("test",  args.data, args.img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return train_loader, test_loader


# =========================
# METRICS
# =========================

def compute_metrics(pred, gt, mask=None):
    if mask is None:
        mask = gt > 0

    p = pred[mask]
    g = gt[mask]

    if p.numel() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "d1": float("nan")}

    abs_rel = ((p - g).abs() / (g + EPSILON)).mean().item()
    rmse = ((p - g).pow(2).mean()).sqrt().item()
    ratio = torch.max(p / (g + EPSILON), g / (p + EPSILON))
    d1 = (ratio < 1.25).float().mean().item()

    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1}


# =========================
# VISUALIZATION
# =========================

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_rgb(t):
    return (t.cpu() * _STD + _MEAN).permute(1, 2, 0).numpy()


def save_visualisation(gen, loader, epoch, save_dir):
    gen.eval()
    device = next(gen.parameters()).device
    batch  = next(iter(loader))

    rgb = batch["rgb"].to(device)
    d1  = batch["depth_1"].to(device)
    d2  = batch["depth_2"].to(device)
    gt  = batch["depth_gt"].to(device)

    with torch.no_grad():
        c1, c2, d_fused = gen(rgb, d1, d2)

    n = min(rgb.shape[0], 4)

    fig = plt.figure(figsize=(21, 3.8 * n))
    gs = gridspec.GridSpec(n, 7, figure=fig)

    for row in range(n):
        def t(x): return x[row, 0].cpu().numpy()

        imgs = [
            denorm_rgb(rgb[row]),
            t(d1), t(d2), t(c1), t(c2), t(d_fused), t(gt)
        ]

        for col, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=None if col == 0 else "magma")
            ax.axis("off")

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"vis_{epoch:03d}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved → {path}")
    gen.train()


# =========================
# TRAIN LOOP
# =========================

def train_one_epoch(gen, disc, opt_g, opt_d, loader, scaler, args, epoch):
    gen.train()
    disc.train()

    device = next(gen.parameters()).device
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    sum_g = sum_d = 0.0

    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device)
        d1  = batch["depth_1"].to(device)
        d2  = batch["depth_2"].to(device)
        gt  = batch["depth_gt"].to(device)

        ones  = torch.ones_like(disc(rgb, gt))
        zeros = torch.zeros_like(ones)

        # ---- Discriminator ----
        opt_d.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, _, fake = gen(rgb, d1, d2)
            loss_d = 0.5 * (
                bce(disc(rgb, gt), ones) +
                bce(disc(rgb, fake.detach()), zeros)
            )
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)
        scaler.update()

        # ---- Generator ----
        opt_g.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            c1, c2, fake = gen(rgb, d1, d2)

            loss = (
                bce(disc(rgb, fake), ones)
                + args.lam_depth * l1(fake, gt)
                + args.lam_smooth * gradient_smoothness_loss(fake)
                + args.lam_sum * confidence_sum_loss(c1, c2)
            )

        scaler.scale(loss).backward()
        scaler.step(opt_g)
        scaler.update()

        sum_g += loss.item()
        sum_d += loss_d.item()

        if (i + 1) % 50 == 0:
            print(f"[{epoch}] step {i+1}/{len(loader)} G={sum_g/(i+1):.4f}")

    return sum_g / len(loader), sum_d / len(loader)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save", default="./outputs")

    parser.add_argument("--lam_depth", type=float, default=10.0)
    parser.add_argument("--lam_smooth", type=float, default=0.1)
    parser.add_argument("--lam_sum", type=float, default=1.0)

    args = parser.parse_args()

    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args)

    gen  = ConfidenceGenerator().to(device)
    disc = PatchDiscriminator().to(device)

    print("Params:", count_params(gen) + count_params(disc))

    opt_g = torch.optim.AdamW(gen.parameters(), lr=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=1e-4)

    scaler = torch.amp.GradScaler("cuda")

    ckpt_dir = os.path.join(args.save, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        g, d = train_one_epoch(gen, disc, opt_g, opt_d, train_loader, scaler, args, epoch)

        print(f"Epoch {epoch} | G={g:.4f} D={d:.4f}")

        if epoch % 5 == 0:
            save_visualisation(gen, test_loader, epoch, args.save)

        if epoch % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "gen": gen.state_dict(),
                "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved → {ckpt_path}")

        # ---- Save best model ----
        if g < best_loss:
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(gen.state_dict(), best_path)
            print(f"Best model updated → {best_path}")


if __name__ == "__main__":
    main()