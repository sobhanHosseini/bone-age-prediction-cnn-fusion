# Bone Age Estimation from Pediatric Hand Radiographs

Lightweight CNNs trained from scratch for bone age prediction under strict no-pretraining constraints. This project investigates uncertainty-aware learning (Deep Label Distribution Learning, DLDL), gender conditioning, and patch-based attention mechanisms on the RSNA Pediatric Bone Age dataset.

---

## 🔎 Project Summary

Bone age assessment from hand radiographs is clinically used to evaluate skeletal maturity and diagnose growth disorders. Manual assessment is time-consuming and subject to inter-observer variability.

This project formulates bone age estimation as a probabilistic prediction task using **Deep Label Distribution Learning (DLDL)** rather than direct regression. All models are trained entirely from scratch.

Key research questions:
- How far can lightweight CNNs go without pretrained backbones?
- Does patch-based attention improve performance under scratch training?
- How much does gender metadata help?
- Does uncertainty-aware learning improve stability?

---

## 🧠 Methods Overview

### 1️⃣ Whole-Image Models (M1, M2)

- Input: resized grayscale radiograph
- Lightweight CNN encoder (Conv → BN → ReLU blocks)
- Global Average Pooling
- Dense(256) → BN → Dropout
- Final Dense(241) logits (age bins 0–240)
- Optional gender embedding concatenated before prediction head

### 2️⃣ Patch-Based Models (M3, M4)

- Fixed overlapping grid extraction
  - Patch size: 128×128
  - Stride: 64
  - 9 patches per image (3×3 grid)
- Shared CNN encoder per patch
- Attention-based fusion:

**M4 — Learned Softmax Attention**
- Scalar importance score per patch
- Softmax normalization across patches
- Weighted sum of embeddings

**M3 — Gated Attention**
- Nonlinear transformation: tanh(Vz) ⊙ sigmoid(Uz)
- Learnable gating before attention scoring
- More expressive patch weighting

---

## 📊 Uncertainty-Aware Learning (DLDL)

Instead of regressing a single age value, we predict a probability distribution over discrete age bins (0–240 months).

For each training sample:
- Construct a Gaussian label distribution centered at the ground-truth age (σ = 4 months)
- Minimize KL divergence between predicted and target distributions
- At inference, compute expectation over predicted probabilities:

  ŷ = Σ p(i) · i

Benefits:
- Models annotation uncertainty
- Provides smoother gradients
- Improves optimization stability

---

## ⚙️ Training Setup

- Optimizer: **AdamW** (decoupled weight decay)
- Learning rate schedule: **Cosine decay**
- Batch size: 32 (reduced for high-resolution experiments)
- Early stopping based on validation MAE
- Mixed precision enabled
- All models trained under identical preprocessing and optimization settings

---

## 🖼 Preprocessing Pipeline

- Resize to 256×256 (or 512×512 for high-resolution study)
- CLAHE (clip limit = 2.0, 8×8 tiles)
- Collimation border masking
- Radiopaque annotation removal (threshold + morphological opening + inpainting)
- Dataset-specific Z-score normalization (computed on training set only, excluding background)

No segmentation or anatomical localization is used.

---

## 📈 Results (Validation Split)

| Model | Resolution | MAE (months) | Params (M) |
|-------|------------|--------------|------------|
| M1 (Image Only) | 256×256 | 12.17 | 0.71 |
| M2 (Image + Gender) | 256×256 | 8.36 | 0.72 |
| M3 (Gated Attention) | 256×256 | 8.81 | 0.79 |
| M4 (Learned Attention) | 256×256 | 9.37 | 0.72 |
| **M2 High-Res** | **512×512** | **7.15** | 2.56 |
| Ensemble (M2+M3+M4) | Mixed | 7.80 | -- |

### Key Findings

- Gender metadata provides the largest single improvement.
- Increasing resolution improves performance more than architectural complexity.
- Patch-based attention improves interpretability but increases inference cost.
- High-resolution whole-image model (M2 512×512) achieves the best single-model trade-off.

---

## 🔬 Error Analysis

- Errors are stable across most pediatric age groups.
- Increased MAE observed at the upper end of the age spectrum (late adolescence).
- Small but consistent gender performance gap.

---

## 📁 Dataset

RSNA Pediatric Bone Age dataset (grayscale hand radiographs with age in months + binary gender).

Official splits:
- Train: 12,611 images
- Validation: 1,425 images

⚠️ Dataset is not redistributed. Please obtain it from the official RSNA/Kaggle source.

---
## 🏁 Final Takeaways

- Under scratch constraints, resolution and data quality matter more than architectural complexity.
- Uncertainty-aware objectives improve stability in noisy medical labeling settings.
- Lightweight CNNs can achieve strong performance without pretrained backbones.
