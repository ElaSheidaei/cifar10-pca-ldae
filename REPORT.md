# CIFAR-10 PCA latent-space DAE (l-DAE-style prototype)

This repository contains a lightweight prototype inspired by **Deconstructing Denoising Diffusion Models for Self-Supervised Learning (Chen et al.)**, I implemented a simplified l-DAE-style prototype adapted to CIFAR-10.
The core idea is to build a **low-dimensional patch-wise tokenizer** with PCA, corrupt samples by adding noise in the **latent space** (with a simple timestep-dependent noise schedule), map back to pixel space, and train a small **denoising autoencoder**. Representation quality is evaluated with a **linear probe** on the encoder features.  
Implementation details are provided as notebooks.

---

## 1) Fast sweep: effect of latent dimension (d = 8, 16, 32)

### Setup (Fast_run.ipynb)
- Dataset: CIFAR-10
- Tokenizer: PCA on **non-overlapping 4×4 patches** (48-D per patch)
- PCA fit: first **5000** training images (patch matrix size: 5000×64 patches)
- Latent corruption: sample a timestep `t ∈ {0,…,T−1}` and add Gaussian noise with
  - `T = 10`, `σ(t)` linearly from `σ_min = 0` to `σ_max = √2`
- Denoiser: small conv autoencoder; features = mean pooled encoder activations (256-D)
- Training budget (fast): **20k train**, **8k test**, **10 AE epochs**, **15 LP epochs**

### Results (linear probe accuracy)
| latent dim d | linear probe acc (%) |
|---:|---:|
| 8  | 37.84 |
| 16 | 37.48 |
| 32 | 35.38 |

The corresponding plot is saved as: `figures/fast_sweep_linear_probe.png`.

### Interpretation (why these numbers make sense)
- **Compression vs task difficulty.** With small `d` (8–16), the PCA tokenizer enforces a stronger information bottleneck, producing smoother/low-frequency patch reconstructions. Under a limited training budget, this can make the denoising task easier and encourages learning more stable features.
- **Higher d can hurt with the same noise schedule and short training.** With `d=32`, the corruption in latent space has more degrees of freedom; after inverse PCA it yields more complex, structured perturbations in pixel space. With only 10 AE epochs, the model may not denoise these effectively, reducing feature quality for linear probing.
- **Small gaps between 8 and 16 are expected** in a short run (few epochs, subset training). The main robust signal in this configuration is the drop at larger latent dimension.

---

## 2) Full run + qualitative visualizations (d = 16)

### Setup (l_DAE_Experiment.ipynb)
- Latent dimension: **d = 16**
- PCA fit: first **5000** training images
- Denoiser training: **full CIFAR-10 train set**, **20 AE epochs**
- Linear probe: **15 epochs**
- Goal: provide qualitative inspection of (i) corruption type and (ii) denoising behavior

### Visualizations
Two figure panels are saved in `figures/`:

1) **Clean vs pixel noise vs latent noise**  
   File: `figures/viz_clean_pixel_latent.png`  
   - Clean image (original CIFAR-10 sample)  
   - Pixel-noised image (baseline i.i.d. pixel corruption)  
   - Latent-noised image (noise added in PCA latent space, then mapped back via inverse PCA)  
   **Observation:** latent-space corruption is visibly more structured and patch-dependent than pure pixel noise.

2) **Clean → latent-noised → denoised output**  
   File: `figures/viz_clean_latent_denoised.png`  
   - Clean image  
   - Latent-noised input  
   - Autoencoder denoised reconstruction  
   

---

## Reproducibility
Open and run the notebooks:
- `notebooks/Fast_run.ipynb`
- `notebooks/l_DAE_Experiment.ipynb`

Recommended environment:
- Python 3.9+
- PyTorch + torchvision
- scikit-learn
- numpy, matplotlib

See `requirements.txt`.

---

## Repository link
GitHub: **<ADD YOUR REPO LINK HERE>**
