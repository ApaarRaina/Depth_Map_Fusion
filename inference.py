import os
import argparse
import numpy as np
from pathlib import Path
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

def normalise(img):
    img=img/img.max()

    return img

def compute_rmse(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    return torch.sqrt(F.mse_loss(pred, gt)).item()


def compute_ssim(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()

    pred = pred
    gt = gt

    return ssim_fn(pred, gt, data_range=1.0)


def compute_psnr(pred, gt):
    pred, gt = normalise(pred), normalise(gt)
    pred = pred.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()

    pred = pred
    gt = gt

    return psnr_fn(gt, pred, data_range=1.0)



def save_depth(depth, path):
    depth = depth.squeeze().cpu().numpy()
    depth = (depth * 255).astype(np.uint8)
    Image.fromarray(depth).save(path)



def run_inference(args):
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    dataset = FusionDataset("test", args.data, args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ConfidenceGenerator().to(device)

    ckpt_path = os.path.join(args.checkpoints, "epoch_30.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(ckpt["gen"])
    model.eval()

    pred_dir = os.path.join(args.out, "pred")
    c1_dir   = os.path.join(args.out, "c1")
    c2_dir   = os.path.join(args.out, "c2")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(c1_dir, exist_ok=True)
    os.makedirs(c2_dir, exist_ok=True)

    total_ssim = 0
    total_psnr = 0
    total_rmse = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            rgb = batch["rgb"].to(device)
            d1  = batch["depth_1"].to(device)
            d2  = batch["depth_2"].to(device)
            gt  = batch["depth_gt"].to(device)
            name = batch["name"][0]

            c1, c2, pred = model(rgb, d1, d2)

            pred = pred

            rmse = compute_rmse(pred, gt)
            ssim = compute_ssim(pred, gt)
            psnr = compute_psnr(pred, gt)

            total_rmse += rmse
            total_ssim += ssim
            total_psnr += psnr
            count += 1

            save_depth(pred, os.path.join(pred_dir, f"{name}.png"))
            save_depth(c1,   os.path.join(c1_dir, f"{name}.png"))
            save_depth(c2,   os.path.join(c2_dir, f"{name}.png"))

    print("\n====== RESULTS ======")
    print(f"SSIM : {total_ssim / count:.4f}")
    print(f"PSNR : {total_psnr / count:.4f}")
    print(f"RMSE : {total_rmse / count:.4f}")


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", default="./inference_outputs")
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
