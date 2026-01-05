# CIFAR-10 PCA latent-space DAE (l-DAE-style)

This repository contains a small CIFAR-10 prototype inspired by latent denoising autoencoder (l-DAE) ideas:
PCA is used as a patch-wise tokenizer, noise is added in latent space with a simple timestep-dependent schedule, and a small denoising autoencoder is trained. Representation quality is evaluated with a linear probe.

## Contents
- `notebooks/Fast_run.ipynb` — fast sweep over latent dimensions (d = 8,16,32) on a reduced dataset
- `notebooks/l_DAE_Experiment.ipynb` — full run for d = 16 + qualitative visualizations
- `figures/` — exported figures from the notebooks
- `results/` — short markdown summary of the sweep

## Quick start (Colab)
Upload the notebook you want (`notebooks/*.ipynb`) to Colab and run all cells.

## Local setup
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then open the notebooks in Jupyter/VSCode.

## Fast sweep results (20k train / 8k test, 10 AE epochs, 15 LP epochs)
| latent dim d | linear probe acc (%) |
|---:|---:|
| 8  | 37.84 |
| 16 | 37.48 |
| 32 | 35.38 |

Plot: `figures/fast_sweep_linear_probe.png`

## License
MIT (see `LICENSE`).
