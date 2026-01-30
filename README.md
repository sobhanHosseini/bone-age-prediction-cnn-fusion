# Bone Age Prediction from Hand Radiographs (CNN Regression + Patch Fusion)

Bone age assessment from pediatric hand radiographs is routinely used to estimate skeletal maturity, but manual atlas-based evaluation is time-consuming and subject to inter-observer variability.  
This project studies bone age estimation as a **supervised regression** task under the constraint of **training CNNs from scratch** (no pretrained backbones). We compare:
- a lightweight **whole-image** CNN regressor
- a **patch-based** framework with feature-level fusion (mean vs learned weighting)
- the impact of **gender metadata** as side information

**Main takeaway:** gender helps a lot; learned patch fusion can improve MAE further, but costs ~10× inference time.

---

## Results (RSNA Validation Split)

| Model | Description | MAE (months) | Params (M) | Inference (ms/img) |
|------:|-------------|-------------:|-----------:|-------------------:|
| M1 | Whole image (no gender) | 15.49 | 0.44 | 0.66 |
| M2 | Whole image + gender | 11.86 | 0.44 | 0.68 |
| M3 | Patches + mean fusion (no gender) | 14.21 | 0.44 | 5.38 |
| M4 | Patches + learned fusion + gender | **10.66** | 0.44 | 5.46 |

---

## Method Overview

### Whole-image regression
1. Resize + normalize image
2. CNN encoder + global average pooling
3. Regression head → bone age (months)
4. Optional: concatenate gender indicator before the regression layers

### Patch-based regression with fusion
1. Resize image
2. Extract **overlapping patches** (fixed grid)
3. Shared CNN encoder processes each patch → patch embeddings `{z_k}`
4. Fuse patch embeddings into a global vector:
   - **Mean fusion:** `z = (1/K) Σ z_k`
   - **Learned fusion:** `α_k = softmax(wᵀ z_k)`, `z = Σ α_k z_k`
5. Optional: concatenate gender after fusion

---

## Dataset

RSNA Pediatric Bone Age (hand X-rays, grayscale) with:
- bone age labels (months)
- binary gender metadata

Official splits used:
- Train: 12,611 images
- Validation: 1,425 images  
(Test labels are not public, so test is not used.)

> You must obtain the dataset through the official RSNA source / Kaggle competition page. This repo does not redistribute the data.

---

## Preprocessing

- Resize images to **256×256**
- Normalize intensities using **train-set mean/std**
- Standardize target age during training:
  - `y' = (y - μ_train) / σ_train`
- Convert predictions back to months for evaluation

No segmentation or anatomical localization is used.

---

## Training

- Optimizer: Adam
- LR: 1e-3
- Batch size: 32
- Epochs: up to 20 with early stopping on validation MAE
- Loss: Huber loss
- Mixed precision enabled (outputs in float32 for stability)

Patch extraction (patch models):
- patch size: **128×128**
- stride: **64**
- fixed number of patches per image
