# Topics on Introduction to Multimedia

Topics: image processing, audio filtering, block-matching motion estimation, vector graphics / 3D mesh rendering, and neural JPEG+noise restoration.

Each assignment is a Jupyter notebook plus the input files it needs. Run notebooks from **that assignment’s folder** so relative paths resolve.

## Repository layout

```text
01-image-enhancement-and-dct/          YCbCr, gamma, histogram eq., JPEG-style DCT
02-audio-filters-and-motion-estimation/ FIR filters + echo; block-matching ME
03-bezier-curves-and-3d-models/        Cubic Bézier curves; Stanford bunny lighting
04-neural-image-restoration/           Residual CNN and UNet on JPEG + Gaussian noise
```

| Path | What it is |
|---|---|
| `*/spec.pdf` | Assignment handout |
| `02-audio-filters-and-motion-estimation/results/` | Sample motion-estimation figures |
| `03-bezier-curves-and-3d-models/results/` | Sample Bézier / bunny figures |
| `**/output/` | Files the notebooks write when you re-run them (gitignored) |
| `04-neural-image-restoration/pytorch-tutorial.md` | PyTorch notes for the restoration assignment |
| `04-neural-image-restoration/manual.md` | AnySchedule + SenseCraft API notes |

## Environment

Use **Python 3.10–3.12**. Python 3.13 is usually fine for assignments 01–03; the restoration assignment depends on PyTorch and is more reliable on 3.10–3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

Register the kernel once so Jupyter can see the venv:

```bash
python -m ipykernel install --user --name cs3570 --display-name "Python (cs3570)"
jupyter notebook
```

In the notebook UI, pick the **Python (cs3570)** kernel.

### Restoration extras (PyTorch)

Install a PyTorch build that matches your GPU / CPU from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r 04-neural-image-restoration/requirements.txt
```

A CUDA GPU is strongly recommended. The notebook also runs on CPU, but the ten training runs will take a long time.

## How to run each assignment

Working directory = the assignment folder. Generated files land in `output/`.

### Image enhancement and DCT compression

```bash
cd 01-image-enhancement-and-dct
jupyter notebook photo_enhancement_and_dct.ipynb
```

Inputs: `scene_dark.jpg`, `Barbara.jpg`.

This notebook still has the assignment `TODO` cells; fill those in, then run all cells. DCT runs write `Barbara_quality-*.jpg` next to the notebook.

### FIR audio filters, downsampling, echo

```bash
cd 02-audio-filters-and-motion-estimation
jupyter notebook fir_audio_filters.ipynb
```

Input: `HW2_Mix.wav`. Writes filtered / downsampled / echo WAVs to `output/`. Convolution on the full mix is the slow step.

### Block-matching motion estimation

```bash
cd 02-audio-filters-and-motion-estimation
jupyter notebook motion_estimation.ipynb
```

Input: `img/000.jpg` … `img/017.jpg`. Full search over the sequence is the slow step. Sample motion-vector / residual figures are already in `results/`.

### Bézier curves and 3D bunny

```bash
cd 03-bezier-curves-and-3d-models
jupyter notebook bezier_and_3d_bunny.ipynb
```

Inputs: `bg.png`, `points.txt`, `bunny.obj`. Part 2 needs Plotly + Kaleido to save `output/2a.png` and `output/2b_*.png`. Interactive `fig.show()` still works if Kaleido is missing. Sample renders are in `results/`.

### Neural image restoration

```bash
cd 04-neural-image-restoration
jupyter notebook jpeg_noise_restoration.ipynb
```

The first dataset cell downloads `train.tar` / `val.tar` from Hugging Face ([`KBlueLeaf/inet-sub`](https://huggingface.co/datasets/KBlueLeaf/inet-sub), ~5 GB) into `./data/`. Re-running is a no-op if that folder already exists.

Training uses `pl.seed_everything(42)`. Do not commit `data/`, checkpoints, or `lightning_logs/`.

Companion docs:

- `pytorch-tutorial.md` — tensors, autograd, the restoration training pieces
- `manual.md` — AnySchedule configs and SenseCraft losses / metrics

## Reproducibility checklist

1. Create a fresh venv with Python 3.10–3.12.
2. `pip install -r requirements.txt` (and `04-neural-image-restoration/requirements.txt` for restoration).
3. Open the notebook **from its own folder**.
4. Run cells top to bottom.
5. Restoration: confirm the seed cell ran (`pl.seed_everything(42)`) before each training run; keep the Hugging Face dataset cache so the split does not get re-downloaded.

Deterministic parts (filters, Bézier, block matching, DCT) do not need a seed. Restoration training on GPU may still differ by a small amount across drivers / CUDA versions even with the seed set.

## Course

National Tsing Hua University, CS3570 Introduction to Multimedia Technology, Spring 2026.
