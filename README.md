# FA-CFG-LDCT: Classifier-Free Guided Diffusion with Frequency-Aware Noise Prediction for Low-Dose CT Denoising

**Final Project — Generative AI Fundamentals (01205596/699)**
**Thinnaphat Deedetch — Kasetsart University**

---

## Overview

This repository contains the implementation of FA-CFG-LDCT, a conditional diffusion model for low-dose CT (LDCT) image denoising. The model incorporates two novel training modifications:

1. **Classifier-Free Guidance (CFG)** — stochastic conditioning dropout during training + guided inference with weight `w`, providing a controllable sharpness-fidelity tradeoff at inference time without retraining.
2. **Frequency-Aware Noise Prediction (FANP)** — augments the standard MSE noise-prediction loss with an FFT-magnitude L1 penalty on the reconstructed clean image estimate, recovering high-frequency anatomical detail suppressed by MSE.

Evaluated on the **Mayo Clinic LDCT Grand Challenge dataset** (D45 sharp kernel), the CFG configuration improves LPIPS by **6.3%** over the baseline (p < 0.001), and FANP improves SSIM by **+0.014** (p < 0.001).

---

## Results Summary

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Raw Input | 34.60 ± 1.87 | 0.7954 | 0.2028 |
| Pix2Pix | 37.59 ± 1.08 | 0.8997 | 0.2598 |
| WGAN-VGG | 34.31 ± 0.81 | 0.7468 | 0.1574 |
| Baseline DDPM | 39.24 ± 1.61 | 0.9002 | 0.1906 |
| **Config B (+CFG)** | **39.54 ± 1.72** | 0.9103 | **0.1782** |
| **Config C (+FANP)** | 39.39 ± 1.60 | **0.9142** | 0.1891 |

---

## Repository Structure

```
FA-CFG-LDCT/
├── README.md
├── requirements.txt
├── download_weights.py       # Script to download pre-trained checkpoints
├── LDCT_CFG_Final.ipynb      # Main training + evaluation notebook (Google Colab)
├── inference.py              # Standalone inference script
└── models/
    └── unet.py               # U-Net architecture definition
```

---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full dependency list. A CUDA-capable GPU is required for training. Inference can run on CPU but will be slow (~60s/image).

---

## Dataset

This project uses the **Mayo Clinic Low-Dose CT Grand Challenge** dataset.

1. Request access at: https://www.aapm.org/GrandChallenge/LowDoseCT/
2. Download the `3mm_D45` scans (FD and QD)
3. Place the zip files at:
   ```
   /content/drive/MyDrive/Tang/FinalGenAI/3mm_D45.zip
   ```
   or update the `ZIP_PATH` variable in the notebook.

---

## Pre-trained Weights

Download pre-trained checkpoints using the provided script:

```bash
python download_weights.py
```

This will download the following checkpoints into `./checkpoints/`:

| File | Config | Description |
|---|---|---|
| `config_A_best.pt` | Baseline | DDPM, MSE loss only |
| `config_B_best.pt` | +CFG | MSE + conditioning dropout, w=1.5 |
| `config_C_best.pt` | +FANP | MSE + FFT spectral penalty |
| `config_D_best.pt` | Proposed | CFG + FANP combined |
| `p2p_best.pt` | Pix2Pix | GAN baseline |
| `wgan_best.pt` | WGAN-VGG | GAN baseline |

---

## Training

Open `LDCT_CFG_Final.ipynb` in **Google Colab** with an A100 GPU runtime.

Set the `SAVE_DIR` variable to your Google Drive path, then run all cells in order. Training all four diffusion configurations takes approximately **10-12 hours** on an A100 40GB GPU.

Key hyperparameters:

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 8 |
| Learning rate | 2e-4 |
| Optimizer | AdamW (weight decay 1e-4) |
| LR schedule | Cosine annealing |
| CFG dropout probability | 0.1 |
| FANP lambda | 0.01 |
| Guidance weight w | 1.5 (inference) |
| Warm-start t* | 100 |
| DDIM steps | 50 |

---

## Inference

```python
import torch
from models.unet import CUNet, Diff

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = Diff(CUNet(), T=1000).to(device)
ckpt = torch.load('checkpoints/config_B_best.pt', map_location=device)
model.load_state_dict(ckpt['m'])
model.eval()

# Run inference (warm-start + CFG guided sampling)
# ldct_tensor: (1, 1, H, W) float tensor in [0, 1]
with torch.no_grad():
    denoised = model.sample_partial(ldct_tensor.to(device), start_step=100)
    # Or with CFG guidance:
    denoised = model.ddim_cfg(ldct_tensor.to(device), steps=50, w=1.5)
```

For full preprocessing (DICOM to tensor), see `inference.py`.

---

## Ablation Study

| Config | CFG | FANP | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| A — Baseline | ✗ | ✗ | 39.24 ± 1.61 | 0.9002 | 0.1906 |
| B — +CFG only | ✓ | ✗ | 39.54 ± 1.72 | 0.9103 | **0.1782** |
| C — +FANP only | ✗ | ✓ | 39.39 ± 1.60 | **0.9142** | 0.1891 |
| D — Proposed | ✓ | ✓ | 39.36 ± 1.62 | 0.9075 | 0.1913 |

All differences vs Config A are significant at p < 0.001 (Wilcoxon signed-rank test).

---

## References

- Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- Nichol & Dhariwal, *Improved DDPM*, ICML 2021
- Song et al., *DDIM*, ICLR 2021
- Ho & Salimans, *Classifier-Free Diffusion Guidance*, NeurIPS Workshop 2022
- Isola et al., *Pix2Pix*, CVPR 2017
- Yang et al., *WGAN-VGG*, IEEE TMI 2018

---

## License

This project is for academic research purposes only.
