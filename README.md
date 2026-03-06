# World Model Visualizer
Train VAE-based world models on visual game environments, watch reconstructions improve in real-time, explore latent spaces, and let the model dream.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red) ![PySide6](https://img.shields.io/badge/gui-PySide6-green)

## Quickstart

```bash
pip install torch pygame PySide6 matplotlib numpy scikit-learn

# Linux only:
sudo apt install libxcb-cursor0

python main.py
```

## What it does

**Dashboard with 5 panels:**

| Panel | Description |
|-------|-------------|
| **Environment** | Play Cosmic Drift (WASD), or watch a random agent |
| **Training** | Collect frames, train the VAE, watch loss curves drop |
| **Reconstruction** | Side-by-side real vs reconstructed frames + MSE/PSNR |
| **Dream** | Model imagines future trajectories from a seed frame |
| **Latent Space** | PCA/t-SNE scatter, click-to-decode, dimension sliders, latent walks |

## Workflow

1. Click **Play** to run the environment (or set policy to Random)
2. Click **Collect 500 Frames** to fill the replay buffer
3. Click **Train** and watch loss curves + reconstructions improve
4. Click **Start Dream** to generate imagined trajectories
5. Explore the **Latent Space** panel -- click points, drag sliders, interpolate

## Project structure

```
environments/    Game environments (Cosmic Drift PoC)
models/          World models (Convolutional VAE + transition MLP)
training/        Trainer, replay buffer
dreaming/        Autoregressive dream sequence generator
gui/             PySide6 dashboard, panels, widgets
utils/           Config, image utils, QThread workers
```

## Extending

Add new environments by subclassing `environments.base.Environment`.
Add new world models by subclassing `models.base.WorldModel`.

## License

MIT
