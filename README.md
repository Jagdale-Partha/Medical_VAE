# Enhanced Context-Encoding Variational Autoencoder (ceVAE+) for Unsupervised Anomaly Detection (UAD) in Brain MRI

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework implementing an **Enhanced Context-Encoding Variational Autoencoder (ceVAE+)** for **Unsupervised Anomaly Detection (UAD)** in 2D axial T2-FLAIR brain MRI scans.

---

## 🔬 Theoretical Anchors & Scientific Motivation

In medical unsupervised anomaly detection (UAD), models are trained exclusively on healthy brain scans to construct a normative distribution $p_{\text{normative}}(x)$. At inference time, pathologies (tumors, lesions, edema) are detected as deviations from this learned healthy anatomy.

This codebase bridges the fundamental trade-offs identified across four foundational works:

1. **Baur et al. (2021) - "Autoencoders for Anomaly Detection in Brain Imaging":**
   Identified the critical dilemma between:
   - **Reconstruction Blur:** Dense 1D bottlenecks compress high frequencies, causing sharp anatomical edges (skull, ventricles, sulci) to have high reconstruction residuals (false positives).
   - **Pathology Leakage:** Spatial skip connections (e.g. U-Nets) allow anomalies to bypass the bottleneck and reconstruct faithfully (false negatives).
   *ceVAE+ Solution:* Employs a dense $16384 \to 128$ bottleneck with **no spatial skip connections**, paired with Instance Normalization and bilinear upsampling to eliminate checkerboard artifacts.

2. **Zimmerer et al. (2019) - "Context-encoding Variational Autoencoder (ceVAE)":**
   Introduced dynamic spatial context inpainting and the dual-scoring mechanism:
   $$\mathcal{A}_{\text{pixel}} = \text{GaussianBlur}_{\sigma=1.5}\left( |x - \hat{x}| \odot \left| \frac{\partial \mathcal{L}_{\text{KL}}}{\partial x} \right| \right)$$
   Multiplying the spatial residual by the input-level KL divergence gradient suppresses benign high-contrast edge residuals and amplifies genuine pathological anomalies.

3. **Pinaya et al. (2021):**
   Targets artifact reduction without the severe compute overhead and latency of multi-stage discrete auto-regressive transformers.

4. **Kumar et al. (2023):**
   Formalizes normative modeling for population-scale outlier detection in neuroimaging.

---

## 📐 Mathematical Formulation

### 1. Training Loss Function
$$\mathcal{L}_{\text{recon}}(x, \hat{x}) = 0.8 \cdot \mathcal{L}_1(x, \hat{x}) + 0.2 \cdot (1 - \text{SSIM}(x, \hat{x}))$$

$$\mathcal{L}_{\text{total}} = 0.5 \left( \mathcal{L}_{\text{recon}}(x, \hat{x}_{\text{clean}}) + \beta \mathcal{L}_{\text{KL}} \right) + 0.5 \mathcal{L}_{\text{recon}}(x, \hat{x}_{\text{inpaint}})$$
where $\beta = 0.001$, and $\mathcal{L}_{\text{KL}} = -\frac{1}{2}\sum (1 + \log \sigma^2 - \mu^2 - \sigma^2)$.

### 2. Dual-Scoring Anomaly Localization
At test time, the input $x$ is passed with autograd tracking enabled:
$$\text{Grad} = \left| \frac{\partial \mathcal{L}_{\text{KL}}}{\partial x} \right|$$
$$\text{Composite} = |x - \hat{x}| \odot \text{Grad}_{\text{norm}}$$
$$\text{AnomalyMap} = \text{GaussianBlur}_{\sigma=1.5}(\text{Composite})$$

---

## 📁 Repository Directory Structure

```text
DL_mini_project_VAE/
├── notebooks/
│   └── ceVAE_plus_UAD_Pipeline.ipynb  # Self-contained Colab/Jupyter notebook
├── src/
│   ├── config.py                      # Hyperparameters, paths & configs
│   ├── data/
│   │   ├── dataset.py                 # Normative splits & Brain MRI phantom
│   │   └── masking.py                 # Dynamic on-the-fly random spatial eraser
│   ├── models/
│   │   ├── layers.py                  # Downsample & Bilinear upsample blocks
│   │   └── cevae.py                   # Encoder, Bottleneck, Decoder, ceVAE+
│   ├── losses/
│   │   ├── ssim.py                    # Differentiable pure PyTorch SSIM
│   │   └── cevae_loss.py              # Composite L1 + SSIM + KL loss engine
│   ├── engine/
│   │   ├── trainer.py                 # Beta-annealing training loop & checkpointing
│   │   └── anomaly_scorer.py          # Input-level autograd KL saliency & dual-scoring
│   ├── metrics/
│   │   └── evaluator.py               # Pixel/Slice AUROC, AUPRC, optimal Dice (⌈Dice⌉)
│   └── visualize/
│       └── plotting.py                # 5-panel clinical diagnostic figures
├── checkpoints/                       # Saved model weights
├── results/                           # Evaluation figures and metrics
├── requirements.txt                   # Environment dependencies
├── train.py                           # CLI training entry point
├── evaluate.py                        # CLI evaluation entry point
├── demo.py                            # Rapid smoke & sanity test (< 5s)
└── README.md                          # Project documentation & research report
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Fast Sanity & Smoke Test
Run the instant self-verification script to test model instantiation, forward/backward autograd, and 5-panel figure generation:
```bash
python demo.py
```

### 3. Model Training
Train ceVAE+ on strictly healthy brain slices:
```bash
python train.py --epochs 30 --batch_size 16 --lr 1e-4 --device cpu
# On GPU:
python train.py --epochs 30 --batch_size 32 --lr 1e-4 --device cuda
```

### 4. Interactive Web Visualization Console
Launch the FastAPI clinical diagnostic web application with full interactive reasoning:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser. Features:
- **Case Gallery:** Quick-load clinical presets (Healthy Controls, Parietal Glioblastoma, Frontal Lobe Neoplasms, Temporal Vasogenic Edema).
- **Patient Upload:** Drag-and-drop support for custom patient MRI slices (`.png`, `.jpg`, `.npy`).
- **5-Stage Pipeline View:** Live inspection of Input ($x$), Normative Recon ($\hat{x}$), Residual ($|x - \hat{x}|$), KL Saliency ($|\nabla_x \mathcal{L}_{\text{KL}}|$), and Diagnostic Overlay.
- **Before/After Split Comparison Wipe:** Interactive swipe slider comparing diseased scan vs. normative reconstruction.
- **High-Res Inspector:** Pathology centroid coordinates $(X, Y)$, bounding box, and lesion burden %.
- **AI Radiologist Reasoning:** Structured 6-stage clinical narrative explaining the mathematical and radiological rationale behind the diagnosis.

### 5. Benchmark & Diagnostic Evaluation
Run quantitative evaluation against pathological slices and generate 5-panel clinical figures:
```bash
python evaluate.py --checkpoint checkpoints/best_cevae_model.pt --num_visualizations 5
```

---

## 📊 5-Panel Clinical Diagnostic Output

The diagnostic engine exports comprehensive 5-panel clinical visualizations:
1. **Input T2-FLAIR ($x$):** Patient axial slice.
2. **Reconstruction ($\hat{x}$):** Normative projection mapping pathological regions back to healthy anatomy.
3. **Residual ($|x - \hat{x}|$):** Pixel-wise difference (often contains edge blur at ventricles/skull).
4. **KL Saliency ($|\nabla_x \mathcal{L}_{\text{KL}}|$):** Gradient indicating which pixels violate the normative latent prior.
5. **Detection Overlay:** Predicted lesion segmentation (Red) vs Ground Truth (Green), with True Positive overlap (Yellow).

---

## ⚠️ Failure Modes & Risk Mitigations

| Failure Mode | Symptom | Underlying Cause | Implemented ceVAE+ Mitigation |
| :--- | :--- | :--- | :--- |
| **Posterior Collapse** | $\mathcal{L}_{\text{KL}} \to 0$, blurry recon | Prior term dominates reconstruction early in training | $\beta$-annealing warmup over first 5 epochs ($\beta=0.001$). |
| **Pathology Leakage** | Lesions copied to $\hat{x}$, low Dice | Spatial skip connections or excessive latent capacity | Dense 1D bottleneck ($16384 \to 128$) with strictly zero skip connections. |
| **Edge False Positives** | High residual along skull/ventricles | Imperfect high-frequency autoencoder reconstruction | Dual-scoring fusion: multiplying residual by input KL gradient $|\nabla_x \mathcal{L}_{\text{KL}}|$. |
| **Vanishing KL Gradients** | Noisy or flat saliency map | Vanishing gradients in deep saturated layers | LeakyReLU(0.2) throughout, InstanceNorm2d, and gradient clipping ($5.0$). |
| **Checkerboard Artifacts** | Grid patterns in output | Transposed convolutions ($4\times4$) | Replaced with Bilinear $2\times$ upsampling followed by $3\times3$ regular convolutions. |
