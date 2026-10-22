# Library Manual — AnySchedule & SenseCraft

Two libraries used in HW4. This doc is grounded in the project sources at:

- AnySchedule — [github.com/KohakuBlueleaf/AnySchedule](https://github.com/KohakuBlueleaf/AnySchedule)
- SenseCraft — [github.com/KohakuBlueleaf/SenseCraft](https://github.com/KohakuBlueleaf/SenseCraft)

Both projects are still under active development; what's covered here is what HW4 actually uses, plus enough context to navigate the source if you want more.

---

## 1. AnySchedule

A composable scheduler. Standard `torch.optim.lr_scheduler` only schedules `lr`; AnySchedule can drive **any numeric attribute on the optimizer's param groups** (`lr`, `weight_decay`, `momentum`, …) with built-in warmup and a mode-based config.

### Constructor

```python
from anyschedule import AnySchedule

sched = AnySchedule(
    optimizer,
    last_epoch=-1,
    config: dict | str | Path = ...,   # dict, or path to a .toml file
)
```

`AnySchedule` subclasses `torch.optim.lr_scheduler.LRScheduler`, so it integrates with anything that already accepts a PyTorch scheduler (Lightning included).

### How it works

- For each top-level key in `config` (e.g. `"lr"`, `"weight_decay"`), AnySchedule builds a sub-scheduler.
- The sub-scheduler is a function of the current step → returns a **factor** (a multiplier).
- On `sched.step()`, the optimizer's attribute for each param group is set to `base_value * factor`, where `base_value` is what was in the optimizer's param group at construction time (e.g. the `lr` you passed to `AdamW`).

So the schedule values in your config are *fractions of peak*, not absolute numbers.

### Modes

The available `mode` values (from `src/anyschedule/scheduler/`):

| `mode` | Shape | Required keys | Optional keys |
|---|---|---|---|
| `"constant"` | constant value during `[start, end)` | — | `value`, `start`, `end`, `warmup`, `min_value` |
| `"cosine"` | cosine decay from `value` → `min_value` | — | `value`, `min_value`, `start`, `end`, `warmup` |
| `"polynomial"` | polynomial decay shaped by `power` | `power` | `value`, `min_value`, `start`, `end`, `warmup` |
| `"step"` | piecewise: multiply by `gamma` every `step_size` steps | `step_size`, `gamma` | `value`, `min_value`, `start`, `end`, `warmup` |
| `"power"` | `value * (1 + step)^(-power)` style decay | `power` | `value`, `min_value`, `start`, `end`, `warmup` |
| `"composer"` | chain several sub-schedules end-to-end | `schedules` (list) | `value`, `end` |

### Common keys

| key | meaning |
|---|---|
| `value` | peak factor for the schedule (default `1.0`) |
| `min_value` | floor factor (default `0.0` for most modes) |
| `start` | first step where the schedule is active (default `0`) |
| `end` | step at which the schedule reaches its terminal value |
| `warmup` | number of steps to linearly ramp from 0 up to `value` before the main mode kicks in |

### Examples

```python
# Cosine warmup + decay to 1% of peak (HW4 default)
sched = AnySchedule(opt, config={"lr": {
    "mode": "cosine", "end": 2000,
    "min_value": 0.01, "warmup": 200,
}})

# Constant after warmup, then drops to 0 abruptly at step 2000
sched = AnySchedule(opt, config={"lr": {
    "mode": "constant", "end": 2000, "warmup": 200,
}})

# Polynomial decay (power=2, i.e. quadratic) to 1% over 2000 steps
sched = AnySchedule(opt, config={"lr": {
    "mode": "polynomial", "end": 2000,
    "min_value": 0.01, "warmup": 200, "power": 2,
}})

# Step decay: × 0.8 every 50 steps
sched = AnySchedule(opt, config={"lr": {
    "mode": "step", "end": 2000,
    "step_size": 50, "gamma": 0.8, "warmup": 200,
}})
```

### Multiple attributes

```python
sched = AnySchedule(opt, config={
    "lr":           {"mode": "cosine",   "end": 2000, "min_value": 0.01, "warmup": 200},
    "weight_decay": {"mode": "constant", "end": 2000},
})
```

Each top-level key is scheduled independently.

### Composer mode (chained schedules)

`composer` lets you build piecewise schedules — useful for things like "warmup-stable-decay" (WSD). Each sub-schedule's `end` may be relative (0–1 fraction of the parent's total) or absolute:

```python
config = {"lr": {
    "mode": "composer",
    "value": 1.0,
    "end": 2000,                 # parent period
    "schedules": [
        {"mode": "constant", "end": 0.25, "warmup": 100, "value": 0.5},
        {"mode": "cosine",   "end": 0.5,  "warmup": 50,  "value": 1.0, "min_value": 0.01},
        {"mode": "polynomial", "end": 0.75, "warmup": 50, "value": 1.0,
         "min_value": 0.01, "power": 2},
        {"mode": "step", "end": 1.0, "warmup": 50, "value": 1.0,
         "min_value": 0.01, "step_size": 50, "gamma": 0.8},
    ],
}}
```

(See `config/example.toml` in the AnySchedule repo for the same example.)

### TOML config

If you'd rather keep the schedule out of code, `config` can also be a path to a TOML file:

```python
sched = AnySchedule(opt, config="schedules/cosine.toml")
```

### Lightning integration

```python
def configure_optimizers(self):
    opt = torch.optim.AdamW(self.parameters(), lr=2e-4)
    sched = AnySchedule(opt, config={"lr": {
        "mode": "cosine", "end": self.total_steps,
        "min_value": 0.01, "warmup": self.warmup_steps,
    }})
    return {
        "optimizer": opt,
        "lr_scheduler": {"scheduler": sched, "interval": "step"},
    }
```

`"interval": "step"` is required — AnySchedule advances per optimizer step, not per epoch.

---

## 2. SenseCraft

Two modules you'll touch in HW4:

- `sensecraft.loss` — a unified loss-mixing framework (`SenseCraftLoss`) plus all the individual loss classes.
- `sensecraft.metrics` — functional evaluation metrics (`psnr`, `ssim`, `ms_ssim`, `lpips`, ...).

### 2a. `SenseCraftLoss`

```python
from sensecraft.loss import SenseCraftLoss

loss_fn = SenseCraftLoss(
    loss_config: list,
    input_range: tuple[float, float] = (-1, 1),
    mode: Literal["2d", "3d"] = "2d",
    return_dict: bool = True,
)
```

#### What it does

- Builds each loss term you list, runs them on the right input range automatically, and weights/sums them for backprop.
- Knows that some losses need `[0, 1]` (SSIM, edge filters), some need `[-1, 1]` (LPIPS, perceptual nets), and some don't care (MSE, L1, FFT). You only declare your data's `input_range` once; SenseCraft converts internally per loss.

#### `loss_config` format

A list. Each entry is either:

- a one-key dict mapping a registered name to its weight, e.g. `{"charbonnier": 1.0}`; or
- a typed config class (see "Typed configs" below) for losses that take extra arguments.

```python
loss_fn = SenseCraftLoss(
    loss_config=[
        {"mse": 1.0},
        {"ssim": 0.1},
        {"lpips": 0.05},
    ],
    input_range=(-1, 1),
)
```

#### Calling it

```python
out = loss_fn(pred, target)            # default return_dict=True
total = out["loss"]                     # weighted sum, what you backprop
mse_term = out["mse"]                   # individual values, for logging
ssim_term = out["ssim"]
```

If you constructed with `return_dict=False`, `out` is just the scalar `total`. You can also override per-call:

```python
total = loss_fn(pred, target, return_dict=False)
```

This is what HW4's `RestorationModule` does — `self.loss = SenseCraftLoss(LOSS_CFG, return_dict=False)`.

#### Weight=0 trick (monitoring without gradient)

A term with `weight=0` is **computed under `no_grad`** and added to the returned dict, but does **not** contribute to `out["loss"]`. Use this to log auxiliary metrics during training without affecting gradients:

```python
loss_fn = SenseCraftLoss([
    {"l1": 1.0},
    {"ssim": 0.0},      # logged, not backpropped
    {"lpips": 0.0},     # logged, not backpropped
], input_range=(-1, 1))
```

#### Available loss names

From the registry (see `SenseCraftLoss.available_losses()` for the live list):

| Category | Names | Required input range |
|---|---|---|
| Basic | `mse`, `l1`, `huber`, `charbonnier` | any |
| Regression | `rmse`, `mape`, `smape`, `log_cosh`, `quantile` | any |
| Robust | `cauchy`, `geman_mcclure`, `welsch`, `tukey`, `wing` | any |
| FFT | `fft`, `patch_fft`, `gaussian_noise` | any |
| Edge | `sobel`, `laplacian`, `canny`, `gradient`, `high_freq`, `multi_scale_gradient`, `structure_tensor` | `[0, 1]` |
| SSIM | `ssim`, `ms_ssim` | `[0, 1]` |
| Perceptual | `lpips`, `convnext`, `dino_convnext`, `dino_vit` | `[-1, 1]` |
| Video / 3D | `ssim3d`, `stssim`, `tssim`, `fdb`, `temporal_accel`, `temporal_fft`, `patch_fft_3d`, `temporal_gradient` | varies (use with `mode="3d"`) |

In HW4 your `input_range=(-1, 1)`, and conversion to whatever each loss needs is handled for you.

#### Typed configs (for losses with extra knobs)

A few losses take more than just a weight. Use the typed config classes:

```python
from sensecraft.loss import (
    SenseCraftLoss, LPIPSConfig, SSIMConfig, PatchFFTConfig,
    DinoV3LossConfig, ConvNextDinoV3LossConfig,
)

loss_fn = SenseCraftLoss(
    loss_config=[
        {"charbonnier": 1.0},
        LPIPSConfig(weight=0.05, net="alex"),       # net: "vgg" | "alex" | "squeeze"
        SSIMConfig(weight=0.1, win_size=11),
        PatchFFTConfig(weight=0.05, patch_size=16),
    ],
    input_range=(-1, 1),
)
```

Most losses don't need this — `{name: weight}` is enough.

### 2b. `sensecraft.metrics` — evaluation API

Separate from the loss module. Returns *quality* values, not loss formulations.

```python
from sensecraft.metrics import psnr, ssim, ms_ssim, rmse, mae, mape, lpips
```

#### Functional API

```python
psnr_db   = psnr(pred, target, data_range=2.0)        # scalar tensor in dB
ssim_val  = ssim(pred, target, data_range=2.0)        # scalar in [-1, 1] (typically [0, 1])
ssim_db   = ssim(pred, target, data_range=2.0, as_db=True)   # -10*log10(1 - ssim) in dB
ms_ssim_v = ms_ssim(pred, target, data_range=2.0)     # MS-SSIM (needs ≥ ~160×160)
rmse_v    = rmse(pred, target)
mae_v     = mae(pred, target)
mape_v    = mape(pred, target)

# LPIPS expects [-1, 1] always; data_range arg doesn't apply.
# Models are auto-cached and auto-moved to the input's device on first call.
lpips_v = lpips(pred, target, net="alex")             # net: "vgg" | "alex" | "squeeze"
```

#### `data_range` reference

Match it to your tensor range:

| Tensor range | `data_range` |
|---|---|
| `[0, 1]` | `1.0` |
| `[-1, 1]` | `2.0` |
| `[0, 255]` | `255.0` |

In HW4, all tensors live in `[-1, 1]`, so use `data_range=2.0`.

#### Metric quick reference

| Function | Range | Better |
|---|---|---|
| `psnr(...)` | dB | higher |
| `ssim(...)` | `[0, 1]` (or dB if `as_db=True`) | higher |
| `ms_ssim(...)` | `[0, 1]` (or dB if `as_db=True`) | higher |
| `rmse(...)`, `mae(...)`, `mape(...)` | ≥ 0 | lower |
| `lpips(...)` | ≥ 0 | lower |

### Tips for the loss ablation (1c / 2c)

- **Use the dict return mode** when comparing mixes — print all terms during training to see which one is actually moving:
  ```python
  loss_fn = SenseCraftLoss(LOSS_CFG, input_range=(-1, 1), return_dict=True)
  out = loss_fn(pred, target)
  for k, v in out.items():
      self.log(f"loss/{k}", v.detach())
  ```
- **Keep total scale comparable across mixes.** A mix that returns a much larger combined loss will have a larger gradient norm, which is effectively a different LR.
- **Look at sample images, not just the number.** Pixel losses (MSE, L1) usually win on PSNR; perceptual losses (LPIPS) usually win visually. They're trading off different things.
- **Heavy losses cost compute.** `lpips` and the DINOv3 perceptual losses run an extra CNN per step. On Colab T4 the slowdown is modest but worth noting in your `1(d)` budgeting.
