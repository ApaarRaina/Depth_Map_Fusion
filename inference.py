import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from GAN_Model import ConfidenceGenerator
from train import FusionDataset

from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def normalise(img):
    img = img / img.max()
    return img


def compute_rmse(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    return torch.sqrt(F.mse_loss(pred, gt)).item()


def compute_ssim(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    pred = pred.squeeze().cpu().numpy()
    gt   = gt.squeeze().cpu().numpy()
    return ssim_fn(pred, gt, data_range=1.0)


def compute_psnr(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    pred = pred.squeeze().cpu().numpy()
    gt   = gt.squeeze().cpu().numpy()
    return psnr_fn(gt, pred, data_range=1.0)


def accumulate_metrics(pred, gt, totals):
    totals["rmse"] += compute_rmse(pred, gt)
    totals["ssim"] += compute_ssim(pred, gt)
    totals["psnr"] += compute_psnr(pred, gt)


def print_results(label, totals, count):
    print(f"\n====== {label} ======")
    print(f"SSIM : {totals['ssim'] / count:.4f}")
    print(f"PSNR : {totals['psnr'] / count:.4f}")
    print(f"RMSE : {totals['rmse'] / count:.4f}")


# ──────────────────────────────────────────────────────────────
# Save helper
# ──────────────────────────────────────────────────────────────

def save_depth(depth_tensor, path):
    depth = depth_tensor.squeeze().cpu().numpy()
    depth = (depth * 255).astype(np.uint8)
    Image.fromarray(depth).save(path)


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def run_inference(args):
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    dataset = FusionDataset("test", args.data, args.img_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # ── Fusion model ──────────────────────────────────────────
    model = ConfidenceGenerator().to(device)
    ckpt_path = os.path.join(args.checkpoints, "epoch_75.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(ckpt["gen"])
    model.eval()

    # ── Output directories ────────────────────────────────────
    dirs = {
        "fusion_pred" : os.path.join(args.out, "fusion", "pred"),
        "fusion_c1"   : os.path.join(args.out, "fusion", "c1"),
        "fusion_c2"   : os.path.join(args.out, "fusion", "c2"),
        "midas"       : os.path.join(args.out, "midas"),          # depth_1
        "dav2"        : os.path.join(args.out, "dav2"),           # depth_2
        "gt"          : os.path.join(args.out, "gt"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # ── Metric accumulators ───────────────────────────────────
    zero = lambda: {"ssim": 0.0, "psnr": 0.0, "rmse": 0.0}
    totals = {
        "fusion" : zero(),
        "midas"  : zero(),
        "dav2"   : zero(),
    }
    count = 0

    # ── Main loop ─────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader):
            rgb  = batch["rgb"].to(device)
            d1   = batch["depth_1"].to(device)   # MiDaS pre-computed output
            d2   = batch["depth_2"].to(device)   # DepthAnythingV2 pre-computed output
            gt   = batch["depth_gt"].to(device)
            name = batch["name"][0]

            # Fusion model forward pass
            c1, c2, pred = model(rgb, d1, d2)

            # Metrics for all three
            accumulate_metrics(pred, gt, totals["fusion"])
            accumulate_metrics(d1,   gt, totals["midas"])
            accumulate_metrics(d2,   gt, totals["dav2"])
            count += 1

            # Save outputs
            save_depth(pred, os.path.join(dirs["fusion_pred"], f"{name}.png"))
            save_depth(c1,   os.path.join(dirs["fusion_c1"],   f"{name}.png"))
            save_depth(c2,   os.path.join(dirs["fusion_c2"],   f"{name}.png"))
            save_depth(d1,   os.path.join(dirs["midas"],       f"{name}.png"))
            save_depth(d2,   os.path.join(dirs["dav2"],        f"{name}.png"))
            save_depth(gt, os.path.join(dirs["gt"],            f"{name}.png"))

    # ── Results ───────────────────────────────────────────────
    print_results("MiDaS (depth_1) vs GT",           totals["midas"],  count)
    print_results("DepthAnythingV2 (depth_2) vs GT", totals["dav2"],   count)
    print_results("Fusion Model vs GT",               totals["fusion"], count)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fusion model + MiDaS + DepthAnythingV2 against GT depth"
    )
    parser.add_argument("--data",        required=True,  help="Path to processed_dataset root")
    parser.add_argument("--checkpoints", required=True,  help="Directory containing epoch_75.pt")
    parser.add_argument("--out",         default="./inference_outputs")
    parser.add_argument("--img_size",    type=int, default=256)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()