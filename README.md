# 🚀 Detail-Centric Depth Map Fusion

Confidence-weighted fusion of monocular depth estimators (**MiDaS** + **Depth Anything v2**) with **GAN-based refinement** for improved depth prediction across indoor and outdoor scenes.

---

## 📌 Overview

Monocular depth estimation models often fail to generalize across diverse environments:

* **MiDaS** → Strong for indoor scenes, struggles outdoors
* **Depth Anything v2** → Robust outdoors, weaker indoors

This project combines their complementary strengths using:

* Learned pixel-wise confidence fusion
* Residual refinement network
* GAN-based training for perceptual quality

👉 Goal: Produce a **single, sharper, and more reliable depth map** than either model individually.

---

## 🧠 Core Idea

We fuse two depth maps using learned confidence weights:

```
D_fused = (c1 * D1 + c2 * D2) / (c1 + c2 + ε)
```

Where:

* `D1`: MiDaS depth
* `D2`: Depth Anything v2 depth
* `c1, c2`: Learned confidence maps

---

## 🏗️ Architecture

### 🔹 ConfidenceGAN

* **Encoder**: MobileNetV3 (feature extraction)
* **Dual Decoder Heads**:

  * Predict confidence maps `c1`, `c2`
* **Fusion Module**:

  * Weighted blending of depth maps
* **Refinement Network**:

  * Residual correction using skip connections

### 🔹 RefinementNet

* Learns residual correction `Δ`
* Final output:

```
D_out = RefineNet(D_fused)
```

* Uses ResBlocks for stable learning

### 🔹 Discriminator

* Patch-based GAN discriminator
* Input: `(RGB + Depth)`
* Encourages local realism

---

## 📂 Dataset Structure

```
processed_dataset/
  train/
    rgb/
    depth_1/
    depth_2/
    depth_gt/
  test/
    rgb/
    depth_1/
    depth_2/
    depth_gt/
```

---

## 📊 Datasets Used

* **NYU Depth v2** → Indoor scenes
* **KITTI** → Outdoor scenes

---

## ⚙️ Data Pipeline

1. Input RGB + Ground Truth
2. Run depth estimators offline
3. Normalize each depth map to `[0,1]`
4. Store processed dataset

✅ Key Design Choice:
Pre-computing depth maps reduces training cost and isolates fusion learning.

---

## 🎯 Training Setup

### Loss Function

| Loss Component      | Description            | Weight |
| ------------------- | ---------------------- | ------ |
| Adversarial (GAN)   | Real vs fake depth     | 1      |
| L1 Depth Loss       | Pixel-wise supervision | 10     |
| Gradient Smoothness | Reduces noise          | 0.1    |
| Confidence Sum Loss | Ensures `c1 + c2 ≈ 1`  | 1      |

---

### ⚡ Training Details

* Optimizer: AdamW
* Learning Rate: `1e-4`
* Epochs: 50
* Batch Size: 4
* Image Size: 256×256
* Mixed Precision (AMP) enabled

---

## 📈 Results

| Model                   | SSIM       | PSNR      | RMSE       |
| ----------------------- | ---------- | --------- | ---------- |
| MiDaS                   | 0.6507     | 13.53     | 0.2239     |
| Depth Anything v2       | 0.3022     | 6.88      | 0.4618     |
| **Fusion Model (Ours)** | **0.7520** | **16.92** | **0.1561** |

---

## ⚠️ Challenges

* GAN Instability
* Confidence Collapse
* Fusion limitations when both inputs fail

---

## ✅ Key Takeaways

* Complementary models improve performance
* Learned fusion is better than fixed blending
* Refinement network improves local errors

---

## 🛠️ Future Work

* Multi-model fusion
* Diffusion-based refinement
* Real-time optimization

---

---

## 🚀 Usage

### 🔹 Dataset Preparation

To download the dataset, run:

```bash
python dataset_download.py
```

Before training, preprocess the dataset to generate the required structure:

```bash
python depth_map_dataset.py
```

This will create the `processed_dataset/` directory in the required format:

```
processed_dataset/
  train/
  test/
```

---

### 🔹 Training

To train the model, run:

```bash
python train.py
```

---

### 🔹 Inference

To run inference on new images, use:

```bash
python inference.py
```

---

## 👤 Author

**Apaar Raina**
