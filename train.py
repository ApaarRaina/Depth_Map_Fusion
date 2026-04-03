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
from GAN_Model import ConfidenceGenerator, PatchDiscriminator, gradient_smoothness_loss, confidence_sum_loss, count_params, EPSILON
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch.hub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class MiDaSIndoor:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True).to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> torch.Tensor:
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp).squeeze().cpu()
        pred = 1.0 / (pred + 1e-6)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return torch.from_numpy(pred.numpy()).unsqueeze(0)  # shape [1,H,W]


class DepthAnythingOutdoor:
    def __init__(self, device):
        self.device = device
        model_id = "depth-anything/Depth-Anything-V2-Large-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        depth = self.model(**inputs).predicted_depth
        depth = F.interpolate(depth.unsqueeze(1),
                              size=(pil.height, pil.width),
                              mode="bicubic",
                              align_corners=False).squeeze(0)
        pred = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return pred.cpu()



class FusionDataset(Dataset):
    def __init__(self, split: str, data_root: str, img_size: int,
                 midas: MiDaSIndoor, outdoor_model: DepthAnythingOutdoor):
        self.img_dir = Path(data_root) / split / "images"
        self.depth_dir = Path(data_root) / split / "depth"  # For ground truth
        self.img_size = (img_size, img_size)
        self.midas = midas
        self.outdoor_model = outdoor_model

        self.samples = sorted([p for p in self.img_dir.glob("*.png") if (self.depth_dir / p.name).exists()])
        if not self.samples:
            raise FileNotFoundError(f"No samples found in {self.img_dir}")

        self.rgb_tf = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        source = "indoor" if path.stem.startswith("indoor") else "outdoor"

        rgb_pil = Image.open(path).convert("RGB")
        rgb_tensor = self.rgb_tf(rgb_pil)

        rgb_np = np.array(rgb_pil)


        if source == "indoor":
            depth_1 = self.midas.predict(rgb_np)
            depth_2 = depth_1.clone()
        else:
            depth_2 = self.outdoor_model.predict(rgb_np)
            depth_1 = depth_2.clone()

        # Ground truth
        raw = np.array(Image.open(self.depth_dir / path.name), dtype=np.float32)
        scale = 1000.0 if source == "indoor" else 256.0
        depth_gt = torch.from_numpy(raw / scale).unsqueeze(0)
        depth_gt = F.interpolate(depth_gt.unsqueeze(0), self.img_size, mode="nearest").squeeze(0)

        return {
            "rgb": rgb_tensor,
            "depth_1": depth_1,
            "depth_2": depth_2,
            "depth_gt": depth_gt,
            "source": source,
            "name": path.stem
        }


def build_loaders(args, midas, outdoor_model):
    print("\n── DataLoaders ──")
    train_ds = FusionDataset("train", args.data, args.img_size, midas, outdoor_model)
    test_ds = FusionDataset("test", args.data, args.img_size, midas, outdoor_model)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, test_loader


def compute_metrics(pred: torch.Tensor,gt:   torch.Tensor,mask: torch.Tensor | None = None):
    if mask is None:
        mask = gt > 0
    p = pred[mask]; g = gt[mask]
    if p.numel() == 0:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "d1": float("nan")}

    abs_rel = ((p - g).abs() / (g + EPSILON)).mean().item()
    rmse    = ((p - g).pow(2).mean()).sqrt().item()
    ratio   = torch.max(p / (g + EPSILON), g / (p + EPSILON))
    d1      = (ratio < 1.25).float().mean().item()
    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1}

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm_rgb(t: torch.Tensor) -> np.ndarray:
    return (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def save_visualisation(gen: nn.Module,
                        loader: DataLoader,
                        epoch: int,
                        save_dir: str):
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

    fig = plt.figure(figsize=(21, 3.8 * n), facecolor="#0d0d0d")
    fig.suptitle(f"Epoch {epoch:03d}  —  Confidence-GAN output",
                 color="white", fontsize=13, fontweight="bold")

    col_titles = ["RGB", "D1 (MiDaS)", "D2 (ZoeDepth)",
                  "C1 (trust D1)", "C2 (trust D2)",
                  "Fused depth", "Ground truth"]
    cmaps = [None, "inferno", "plasma", "RdYlGn", "RdYlGn", "magma", "magma"]

    gs = gridspec.GridSpec(n, 7, figure=fig, wspace=0.03, hspace=0.18)

    for row in range(n):
        def t(x): return x[row, 0].cpu().numpy()

        imgs = [
            denorm_rgb(rgb[row]),
            t(d1), t(d2), t(c1), t(c2), t(d_fused), t(gt),
        ]
        src = batch["source"][row]
        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, cmap=cmap)
            if row == 0:
                ax.set_title(col_titles[col], color="white",
                             fontsize=8, fontweight="bold", pad=3)
            ax.axis("off")
            border = "#4fc3f7" if src == "indoor" else "#a5d6a7"
            for sp in ax.spines.values():
                sp.set_edgecolor(border); sp.set_linewidth(1.2)

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"vis_epoch_{epoch:03d}.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved visualisation → {out_path}")
    gen.train()

def train_one_epoch(gen, disc, opt_g, opt_d,
                    loader, scaler, args, epoch: int):
    gen.train(); disc.train()
    device = next(gen.parameters()).device

    bce  = nn.BCEWithLogitsLoss()
    l1   = nn.L1Loss()

    sum_g = sum_d = 0.0

    for i, batch in enumerate(loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        d1  = batch["depth_1"].to(device, non_blocking=True)
        d2  = batch["depth_2"].to(device, non_blocking=True)
        gt  = batch["depth_gt"].to(device, non_blocking=True)

        with torch.no_grad():
            patch_shape = disc(rgb, gt).shape
        ones  = torch.ones (patch_shape, device=device)
        zeros = torch.zeros(patch_shape, device=device)

        opt_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            _, _, d_fused = gen(rgb, d1, d2)
            loss_d = 0.5 * (
                bce(disc(rgb, gt),              ones)  +
                bce(disc(rgb, d_fused.detach()), zeros)
            )
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)
        scaler.update()

        opt_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            c1, c2, d_fused = gen(rgb, d1, d2)

            loss_adv    = bce(disc(rgb, d_fused), ones)
            loss_depth  = l1(d_fused, gt)
            loss_smooth = gradient_smoothness_loss(d_fused)
            loss_csum   = confidence_sum_loss(c1, c2)

            loss_g = (loss_adv
                      + args.lam_depth  * loss_depth
                      + args.lam_smooth * loss_smooth
                      + args.lam_sum    * loss_csum)

        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        sum_g += loss_g.item()
        sum_d += loss_d.item()

        if (i + 1) % max(1, len(loader) // 4) == 0:
            print(f"  [{epoch:03d}] step {i+1:4d}/{len(loader)}  "
                  f"G={sum_g/(i+1):.4f}  D={sum_d/(i+1):.4f}")

    return sum_g / len(loader), sum_d / len(loader)

@torch.no_grad()
def evaluate(gen, loader):
    gen.eval()
    device = next(gen.parameters()).device

    all_abs, all_rmse, all_d1 = [], [], []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        d1  = batch["depth_1"].to(device, non_blocking=True)
        d2  = batch["depth_2"].to(device, non_blocking=True)
        gt  = batch["depth_gt"].to(device, non_blocking=True)

        _, _, fused = gen(rgb, d1, d2)
        m = compute_metrics(fused, gt)
        all_abs.append(m["abs_rel"])
        all_rmse.append(m["rmse"])
        all_d1.append(m["d1"])

    gen.train()
    return {
        "abs_rel": np.nanmean(all_abs),
        "rmse":    np.nanmean(all_rmse),
        "d1":      np.nanmean(all_d1),
    }

def save_checkpoint(path, epoch, gen, disc, opt_g, opt_d,sched_g, sched_d, best_rmse):
    torch.save({
        "epoch":     epoch,
        "gen":       gen.state_dict(),
        "disc":      disc.state_dict(),
        "opt_g":     opt_g.state_dict(),
        "opt_d":     opt_d.state_dict(),
        "sched_g":   sched_g.state_dict(),
        "sched_d":   sched_d.state_dict(),
        "best_rmse": best_rmse,
    }, path)


def load_checkpoint(path, gen, disc, opt_g, opt_d, sched_g, sched_d):
    ckpt = torch.load(path, map_location="cpu")
    gen.load_state_dict(ckpt["gen"])
    disc.load_state_dict(ckpt["disc"])
    opt_g.load_state_dict(ckpt["opt_g"])
    opt_d.load_state_dict(ckpt["opt_d"])
    sched_g.load_state_dict(ckpt["sched_g"])
    sched_d.load_state_dict(ckpt["sched_d"])
    print(f"  Resumed from epoch {ckpt['epoch']}  "
          f"(best RMSE so far: {ckpt['best_rmse']:.4f})")
    return ckpt["epoch"], ckpt["best_rmse"]


def main():
    args = parse_args()
    os.makedirs(args.save, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  Confidence-GAN — Training Pipeline")
    print(f"  Device : {device}")
    print(f"  Epochs : {args.epochs}  |  Batch : {args.batch}  |  "
          f"Img : {args.img_size}²")
    print("=" * 60)

    train_loader, test_loader = build_loaders(args)

    gen  = ConfidenceGenerator().to(device)
    disc = PatchDiscriminator().to(device)

    print(f"\n  Generator params      : {count_params(gen):>10,}")
    print(f"  Discriminator params  : {count_params(disc):>10,}")
    print(f"  Total                 : {count_params(gen)+count_params(disc):>10,}")

    opt_g = torch.optim.AdamW(gen.parameters(),
                               lr=args.lr_g, betas=(0.5, 0.999),
                               weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(),
                               lr=args.lr_d, betas=(0.5, 0.999),
                               weight_decay=1e-4)

    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_d, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_rmse   = float("inf")
    if args.resume:
        start_epoch, best_rmse = load_checkpoint(
            args.resume, gen, disc, opt_g, opt_d, sched_g, sched_d
        )
        start_epoch += 1

    print("\n── Initial output (before training) ──")
    save_visualisation(gen, test_loader, epoch=0, save_dir=args.save)

    print(f"\n── Training for {args.epochs} epochs ──")
    history = {"g_loss": [], "d_loss": [], "rmse": [], "abs_rel": [], "d1": []}

    for epoch in range(start_epoch, args.epochs + 1):
        g_loss, d_loss = train_one_epoch(
            gen, disc, opt_g, opt_d, train_loader, scaler, args, epoch
        )
        sched_g.step()
        sched_d.step()

        metrics = evaluate(gen, test_loader)

        history["g_loss"].append(g_loss)
        history["d_loss"].append(d_loss)
        history["rmse"].append(metrics["rmse"])
        history["abs_rel"].append(metrics["abs_rel"])
        history["d1"].append(metrics["d1"])

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"G={g_loss:.4f}  D={d_loss:.4f}  "
              f"RMSE={metrics['rmse']:.4f}  "
              f"AbsRel={metrics['abs_rel']:.4f}  "
              f"δ<1.25={metrics['d1']:.4f}")

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            save_checkpoint(
                os.path.join(args.save, "best.pt"),
                epoch, gen, disc, opt_g, opt_d, sched_g, sched_d, best_rmse,
            )
            print(f"  ★ New best RMSE={best_rmse:.4f}  →  saved best.pt")

        if epoch % args.vis_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                ckpt_path, epoch, gen, disc,
                opt_g, opt_d, sched_g, sched_d, best_rmse,
            )
            save_visualisation(gen, test_loader, epoch, args.save)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d0d0d")
    ep = range(1, len(history["g_loss"]) + 1)

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#444")

    axes[0].plot(ep, history["g_loss"], color="#7F77DD", label="G loss")
    axes[0].plot(ep, history["d_loss"], color="#D85A30", label="D loss")
    axes[0].set_title("GAN losses", color="white"); axes[0].legend()

    axes[1].plot(ep, history["rmse"], color="#1D9E75")
    axes[1].set_title("RMSE (↓)", color="white")

    axes[2].plot(ep, history["d1"], color="#EF9F27")
    axes[2].set_title("δ<1.25 accuracy (↑)", color="white")

    plt.tight_layout()
    curve_path = os.path.join(args.save, "training_curves.png")
    plt.savefig(curve_path, dpi=110, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)

    print(f"\nTraining complete.")
    print(f"   Best RMSE : {best_rmse:.4f}")
    print(f"   Curves    : {curve_path}")
    print(f"   Best ckpt : {os.path.join(args.save, 'best.pt')}")


if __name__ == "__main__":
    main()