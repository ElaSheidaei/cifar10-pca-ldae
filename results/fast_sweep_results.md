# Fast sweep results (CIFAR-10)

Settings: 20k train / 8k test, 10 AE epochs, 15 LP epochs, PCA fit on 5000 images, 4×4 patches.

| latent dim d | linear probe acc (%) |
|---:|---:|
| 8  | 37.84 |
| 16 | 37.48 |
| 32 | 35.38 |

Main observation: performance is similar for d=8 and d=16 in this short run, while d=32 performs worse under the same noise schedule and training budget.
