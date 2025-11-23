# GPT-Creating-model
In this exercise, we build a small version of a GPT (Generative Pre-trained Transformer) model completely from scratch using PyTorch. 
# GPT Creating Model — Minimal Reproducible Example

This folder contains a minimal GPT-style Transformer implementation and a small training script to reproduce a toy GPT model. It's intended for learning and experimenting with hyperparameters: number of layers, number of heads, and embedding size.

Files:
- `model.py` — lightweight GPT model (PyTorch)
- `data.py` — tiny text loader and simple character-level tokenizer
- `train.py` — training loop with CLI flags for hyperparameters
- `sample_text.txt` — small dataset (Shakespeare-ish snippet)
- `requirements.txt` — Python package list

Quick start (PowerShell):

```powershell
# Create virtual env (optional)
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# Install PyTorch according to your CUDA/OS (see https://pytorch.org )
pip install -r requirements.txt

# Train a tiny model (low memory)
python train.py --epochs 50 --batch_size 32 --block_size 64 --n_layers 4 --n_heads 4 --embed_size 256

# Larger model (more RAM/GPU)
python train.py --epochs 50 --batch_size 16 --block_size 128 --n_layers 8 --n_heads 8 --embed_size 512
```

New options
- `--checkpoint_every N`: save a checkpoint every N steps (useful for long runs). Default `0` (only final save).
- `--resume PATH`: resume training from a checkpoint file saved previously.
- `--amp`: enable mixed-precision training (uses `torch.cuda.amp` when CUDA available) to reduce memory usage and speed up training on GPUs.

Example (resume and AMP):

```powershell
python train.py --epochs 100 --batch_size 8 --block_size 128 --n_layers 8 --n_heads 8 --embed_size 512 --checkpoint_every 500 --amp
# If the run is interrupted, resume with:
python train.py --resume checkpoint.pt.step500.pt --amp
```

Notes:
- If you have a CUDA-capable GPU, install the matching `torch` build from https://pytorch.org and `train.py` will use it automatically.
- Increasing `embed_size`, `n_layers`, or `n_heads` increases memory usage quadratically for attention-heavy models; monitor GPU/host RAM.

Tips for experimentation:
- Try `--n_heads` divisors of `--embed_size` (e.g., embed 512 with 8 heads -> 64-dim per head).
- Increase `--block_size` to let the model see longer contexts.
- When you change `embed_size` or `n_heads`, keep `embed_size % n_heads == 0`.

