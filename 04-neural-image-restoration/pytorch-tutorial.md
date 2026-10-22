# PyTorch from Scratch — A Tutorial for HW4

This tutorial builds up the parts of PyTorch you actually need for HW4, from the bottom. If you already know tensors, skip to §3 (Autograd). If you've trained models in PyTorch before, skip to §6 (the HW4-specific bits).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 1. Tensors are values

A `torch.Tensor` is a multi-dimensional array — for most purposes you can pretend it's a Python number. It supports the same arithmetic, only it works element-wise on N-D arrays at GPU speed.

### Creating tensors

```python
torch.tensor(3.14)             # scalar (0-D)
torch.tensor([1., 2., 3.])     # 1-D, shape (3,)
torch.tensor([[1, 2], [3, 4]]) # 2-D, shape (2, 2)

torch.zeros(2, 3)              # shape (2, 3), filled with 0
torch.ones(4)                  # shape (4,),    filled with 1
torch.full((2, 2), 0.5)        # shape (2, 2),  filled with 0.5
torch.arange(0, 10)            # 1-D: 0, 1, ..., 9
torch.linspace(-1, 1, 5)       # 1-D: -1.0, -0.5, 0.0, 0.5, 1.0
torch.randn(3, 256, 256)       # random normal, shape (3, 256, 256)
torch.rand(8)                  # uniform in [0, 1)
```

Two pieces of "tensor metadata" matter:

- **`.shape`** — the size along each dim. `t.shape` returns a `torch.Size`; you can also index it: `t.shape[0]`.
- **`.dtype`** — element type: `torch.float32` (default), `torch.float16`, `torch.int64`, `torch.bool`, …
- **`.device`** — `cpu`, `cuda:0`, etc. Move with `t = t.to("cuda")`.

```python
t = torch.randn(2, 3)
print(t.shape, t.dtype, t.device)  # torch.Size([2, 3]) torch.float32 cpu
```

### Element-wise operations — treat tensors like numbers

Every operator you'd write on a Python number works element-wise on a tensor (or pair of tensors of compatible shapes):

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

a + b      # tensor([5., 7., 9.])
a * b      # tensor([ 4., 10., 18.])
a ** 2     # tensor([1., 4., 9.])
a / 2      # tensor([0.5000, 1.0000, 1.5000])
torch.sqrt(a)
torch.exp(a)
torch.log(a)
a.sin(); a.cos(); a.tanh()    # method form
a.clamp(0, 2)                  # element-wise clip to [0, 2]
```

Element-wise comparison gives a bool tensor:
```python
(a > 1.5)             # tensor([False,  True,  True])
((a > 1.5) & (a < 3)) # element-wise and
```

Reductions collapse a tensor along a dim (or all dims):
```python
a.sum()              # scalar tensor (sum of all elements)
a.mean()             # scalar tensor
a.max(); a.min()
m = torch.randn(4, 5)
m.sum(dim=0)         # shape (5,)  — collapse rows
m.sum(dim=1)         # shape (4,)  — collapse cols
m.mean(dim=1, keepdim=True)  # shape (4, 1) — keeps the dim with size 1
```

### Broadcasting

When two tensors have different shapes, PyTorch automatically aligns them by adding size-1 dims on the left. As long as each non-matching dim is `1` somewhere, the operation works:

```python
img = torch.randn(3, 256, 256)        # (C, H, W)
mean = torch.tensor([0.5, 0.5, 0.5])  # (3,)
mean = mean.view(3, 1, 1)             # (3, 1, 1) — now broadcast-compatible
normed = (img - mean) / 0.5           # OK, broadcasts to (3, 256, 256)
```

Rule of thumb: when two tensors don't have the same rank, **`view` / `reshape` to add the size-1 dims yourself** — it's clearer and you'll catch shape bugs earlier.

### Reshape, view, permute, transpose

These don't move any memory in most cases — they just change how the same numbers are interpreted.

```python
x = torch.arange(12)        # (12,)
x.view(3, 4)                # (3, 4)        same data, new shape
x.view(2, 2, 3)             # (2, 2, 3)
x.reshape(4, 3)             # like view but also works on non-contiguous tensors

img = torch.randn(2, 3, 256, 256)   # (B, C, H, W)
img.permute(0, 2, 3, 1)             # (B, H, W, C)
img.transpose(1, 2)                 # (B, H, C, W) — swap dims 1 and 2

img.flatten(start_dim=1)            # (2, 3*256*256)  — flatten everything from dim 1 on
```

`squeeze` / `unsqueeze` remove / add size-1 dims:
```python
t = torch.zeros(1, 4, 1)
t.squeeze()           # shape (4,)
t.squeeze(0)          # shape (4, 1) — only squeezes dim 0
t.squeeze().unsqueeze(0)  # back to (1, 4)
```

---

## 2. Linear algebra

### Matrix multiplication

`@` (or `torch.matmul`) is the matmul operator:
```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B            # (3, 5)
```

For batched matmul, PyTorch treats leading dims as batch:
```python
A = torch.randn(8, 3, 4)   # 8 matrices of shape (3, 4)
B = torch.randn(8, 4, 5)
C = A @ B                  # (8, 3, 5)
```

For "matrix × vector", make the vector a column or row:
```python
W = torch.randn(10, 4)
x = torch.randn(4)
W @ x        # shape (10,)  — works directly
```

### `einsum` — index notation

`einsum` lets you write arbitrary contractions by naming the axes. Read it as: "for each output index, sum over the indices that don't appear in the output".

```python
# matmul: i,k * k,j -> i,j
torch.einsum("ik,kj->ij", A, B)

# batched matmul
torch.einsum("bik,bkj->bij", A, B)

# inner product of two vectors
torch.einsum("i,i->", v1, v2)

# row-wise sum: m has shape (M, N), sum over N -> (M,)
torch.einsum("mn->m", m)

# attention scores: q (B, H, T, D), k (B, H, T, D) -> (B, H, T, T)
torch.einsum("bhqd,bhkd->bhqk", q, k)
```

If the index is not in the output, it's contracted (summed). If it is, it's preserved. einsum is the most readable way to write any "what dim goes where" operation that isn't a plain matmul.

### Convolution

For HW4, convolutions are wrapped by `nn.Conv2d` (next section). Under the hood, `F.conv2d(input, weight, bias=None, stride=1, padding=0)` does the math:

```python
img = torch.randn(2, 3, 256, 256)        # (B, C_in, H, W)
w   = torch.randn(16, 3, 3, 3)           # (C_out, C_in, kH, kW)
out = F.conv2d(img, w, padding=1)        # (2, 16, 256, 256)
```

---

## 3. Autograd: the engine that trains everything

PyTorch tracks every operation you do on a tensor that has `requires_grad=True`, and computes derivatives automatically. This is **the** thing that makes deep learning work.

### Toy example: fit `a * b = 1`

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a * b              # forward: c = 6.0
loss = (c - 1.0) ** 2  # we want c == 1, so loss = 25.0

loss.backward()        # computes d(loss)/d(a) and d(loss)/d(b) automatically

print(a.grad)  # tensor(30.)   — derivative of (a*b - 1)^2 w.r.t. a, evaluated at (2,3)
print(b.grad)  # tensor(20.)
```

What just happened:
1. The forward pass built a small graph: `a, b → mul → c → sub → square → loss`.
2. `loss.backward()` walked the graph backward and applied the chain rule.
3. The gradient of `loss` w.r.t. each leaf tensor is now in `.grad`.

To manually do gradient descent on this:
```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
lr = 0.05

for step in range(50):
    c = a * b
    loss = (c - 1.0) ** 2

    loss.backward()           # accumulate grads into .grad
    with torch.no_grad():     # don't track this update in the graph
        a -= lr * a.grad
        b -= lr * b.grad
        a.grad.zero_()        # reset for next step
        b.grad.zero_()

print(a.item(), b.item(), (a * b).item())
# something like 0.41, 2.45, 1.00 — a*b ≈ 1
```

The minimization landscape isn't unique — `a*b=1` has infinitely many solutions, so you'll converge to whichever one is closest to your initialization. That's fine.

### Three things to remember

1. **`.backward()` accumulates** into `.grad`. Always reset (`grad.zero_()` or `optimizer.zero_grad()`) before the next backward, or your gradients will pile up across steps.
2. **`with torch.no_grad():`** turns off the graph for everything inside the block. Use it for parameter updates and for evaluation — anything that doesn't need to backprop.
3. **`tensor.detach()`** returns a copy with `requires_grad=False`. Use it when you want the *value* of a tensor but not its gradient connection. (E.g. `D(G(x).detach())` when training a GAN's discriminator.)

---

## 4. `nn.Module` — bundles of parameters

A neural network is a function with thousands or millions of learnable numbers. `nn.Module` is the bookkeeping class: anything you assign to `self.<name>` that's a `nn.Parameter`, another `Module`, or a list/dict of those, is automatically tracked.

### Built-in modules

The library ships every common layer:

```python
linear = nn.Linear(in_features=128, out_features=10)
conv   = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
norm   = nn.GroupNorm(num_groups=8, num_channels=16)
act    = nn.SiLU()              # Swish: x * sigmoid(x)
drop   = nn.Dropout(p=0.1)
```

All of them are *callable* — you use them like functions. The forward pass shapes that matter for HW4:

```python
x = torch.randn(2, 3, 256, 256)
y = conv(x)         # (2, 16, 256, 256) — channels go from 3 -> 16, padding=1 keeps H, W
```

Stacking layers: `nn.Sequential` runs them in order.
```python
block = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.SiLU(),
    nn.Conv2d(16, 16, 3, padding=1),
)
y = block(x)
```

### Parameters and `.parameters()`

A `Module` knows its trainable tensors. You feed `.parameters()` to an optimizer:

```python
print(sum(p.numel() for p in conv.parameters()))   # total number of learnable scalars
```

`.to(device)` recursively moves every parameter:
```python
model = block.to("cuda")
```

### Custom modules

When the layer you want isn't a single built-in, subclass `nn.Module`. Three rules:

1. Call `super().__init__()` first.
2. Register submodules and parameters in `__init__`.
3. Define the forward computation in `forward`.

```python
class ResBlock(nn.Module):
    """y = x + W2(SiLU(W1(x)))"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=p)
        self.act   = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=p)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))
```

This is exactly the block you implement in HW4 Part 1.

### Composition

Modules can hold other modules — that's how networks are built up:

```python
class TinyCNN(nn.Module):
    def __init__(self, num_blocks=4, hidden=32):
        super().__init__()
        self.conv_in  = nn.Conv2d(3, hidden, 3, padding=1)
        self.blocks   = nn.Sequential(*[ResBlock(hidden) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(hidden, 3, 3, padding=1)

    def forward(self, x):
        feat = self.blocks(self.conv_in(x))
        return self.conv_out(feat)
```

**Recursion is allowed.** A module can construct another instance of *the same class* in its `__init__` — that's how the recursive `UNetStage` in Part 2 works:
```python
class UNetStage(nn.Module):
    def __init__(self, ch, depth):
        super().__init__()
        # ... encoder / decoder layers ...
        self.mid = UNetStage(ch, depth - 1) if depth > 1 else nn.Identity()
```

`nn.Identity()` is a "do-nothing" module — it returns its input unchanged. Useful as a placeholder so `forward` doesn't need an `if`.

---

## 5. A formal training loop

Putting it together: tensors → autograd → module → optimizer → data → train.

```python
model = TinyCNN(num_blocks=6, hidden=32).to("cuda")

opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

for batch_idx, (degraded, clean) in enumerate(train_loader):
    degraded = degraded.to("cuda")
    clean    = clean.to("cuda")

    pred = model(degraded)                    # forward
    loss = F.mse_loss(pred, clean)            # scalar tensor

    opt.zero_grad(set_to_none=True)           # clear last step's grads
    loss.backward()                           # autograd computes parameter grads
    opt.step()                                # AdamW updates each parameter
```

Six pieces, every PyTorch training script has them:

1. **Model** — an `nn.Module` whose `.parameters()` are tracked.
2. **Optimizer** — wraps the parameters and knows how to update them given `.grad`. `AdamW` is fine for almost everything in 2026.
3. **DataLoader** — yields `(input, target)` batches. We've built one for you in HW4.
4. **Forward** — call `model(...)` to compute the prediction.
5. **Loss** — a scalar tensor that measures "how wrong". Anything differentiable is allowed.
6. **Backward + step** — `loss.backward()` to compute grads, `opt.step()` to update params, `opt.zero_grad()` for the next round.

### LR schedules

The learning rate is a single number that often needs to change *over time* — start small (warmup), peak in the middle, anneal to small at the end (cosine). HW4 uses `AnySchedule`:

```python
from anyschedule import AnySchedule
sched = AnySchedule(opt, config={"lr": {
    "mode": "cosine", "end": 2000,       # 2000 steps total
    "min_value": 0.01,                   # end at 1% of peak
    "warmup": 200,                       # ramp from 0 -> peak in 200 steps
}})

for ...:
    ...
    opt.step()
    sched.step()                         # advance the schedule one step
```

### Lightning removes the boilerplate

PyTorch Lightning wraps the loop above so you don't have to type `opt.zero_grad / loss.backward / opt.step` for every project. You define `training_step` and `configure_optimizers`; Lightning calls them.

```python
import lightning.pytorch as pl

class Module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4)

    def training_step(self, batch, batch_idx):
        deg, clean = batch
        pred = self.model(deg)
        loss = F.mse_loss(pred, clean)
        self.log("loss", loss, prog_bar=True)
        return loss        # Lightning does .backward() and .step() for you

    def validation_step(self, batch, batch_idx):
        deg, clean = batch
        pred = self.model(deg).clamp(-1, 1)
        self.log("val_psnr", psnr(pred, clean), prog_bar=True)

trainer = pl.Trainer(max_steps=2000, accelerator="gpu", precision="16-mixed")
trainer.fit(my_module, train_loader, val_loader)
```

`precision="16-mixed"` runs the forward / backward in fp16 with fp32 master weights — typically 1.5–2× faster on a modern GPU.

---

## 6. HW4-specific bits

This section is the stuff you'll actually use in the homework — it depends on §1–§5 above.

### 6a. Image normalization to `[-1, 1]`

The dataset converts every image to a tensor in `[-1, 1]` (via `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`). A symmetric range plays well with SiLU/Tanh activations and matches what LPIPS expects.

When computing PSNR, the `data_range` argument has to match the tensor range. For `[-1, 1]`, `data_range=2.0`:
```python
def psnr(pred, target, data_range=2.0):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(torch.tensor(data_range ** 2) / mse)
```

For *display*, run `denormalize(x) = x * 0.5 + 0.5` to get back to `[0, 1]`, then clamp.

### 6b. Residual blocks

Already covered above. Recap of HW4 conventions:

- Each `ResBlock` is `y = x + f(x)`.
- Use **SiLU** or **Mish**, not ReLU/GELU.

The model goes from `degraded` to a clean prediction directly — there's nothing magical about how the input maps to the output, just a stack of layers. Some restoration papers add a *global* skip on the input (predicting a residual `delta` such that the output is `input + delta`) and zero-init the output conv so training starts at the identity. This is one technique in your toolbox; whether it helps here is something you can measure.

### 6c. Multi-scale: `PixelUnshuffle` / `PixelShuffle`

```
PixelUnshuffle(2):  (B, C, 2H, 2W)  ->  (B, 4C, H, W)    # space -> channels
PixelShuffle(2):    (B, 4C, H, W)   ->  (B, C, 2H, 2W)   # channels -> space
```

Pure permutations — no information is lost on the down step, no checkerboard on the up step. Pair with a 1×1 conv to keep the channel count constant:

```python
self.down = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4*C, C, 1), nn.SiLU())
self.up   = nn.Sequential(nn.Conv2d(C, 4*C, 1), nn.SiLU(), nn.PixelShuffle(2))
```

You're allowed to swap these for strided / transpose convs in your "free to explore" sections.

### 6d. Skip connections (concat-style)

Channels stack along `dim=1`, then a 1×1 conv mixes them:
```python
merged = torch.cat([x, skip], dim=1)   # (B, 2C, H, W)
out    = self.merge(merged)            # 1x1 Conv 2C -> C
```

### 6e. Loss combinations via SenseCraft

A weighted sum of losses is just `a * L1 + b * L2 + ...`. SenseCraft wraps that:
```python
from sensecraft.loss import SenseCraftLoss
loss = SenseCraftLoss([{"mse": 1.0}, {"ssim": 0.1}], return_dict=False)
total = loss(pred, target)
```

For the loss-ablation section: try pure `[{"mse": 1.0}]`, then `[{"l1": 1.0}, {"ssim": 0.1}]`, then `[{"mse": 1.0}, {"lpips": 0.05}]` and compare. Pixel losses (MSE/L1) tend toward higher PSNR but blurrier output; perceptual losses (LPIPS) trade some PSNR for sharpness.

### 6f. `DataLoader` config

```python
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,             # CPU processes that decode + JPEG-degrade
    persistent_workers=True,   # keep them alive between epochs
    drop_last=True,            # drop the trailing partial batch
)
```

`num_workers > 0` matters here because the dataset's per-sample JPEG-encode + noise step is CPU work; without parallel workers your GPU starves between batches. On Colab (2 vCPUs) `num_workers=2` is usually optimal; locally pick 4–8.

---

## 7. Common pitfalls

1. **Shape mismatch on `+`** — if `x.shape != f(x).shape`, `x + f(x)` errors out. Padding must be `kernel_size // 2` for stride=1.
2. **Forgot `super().__init__()`** in your `nn.Module` → `'Module' object has no attribute '_parameters'`.
3. **Forgot `opt.zero_grad()`** before `loss.backward()` → grads accumulate and your loss explodes.
4. **`tensor.item()` on a non-scalar** raises. Use `.mean().item()` for logging.
5. **`val_psnr` keeps falling on its own** — your network is diverging. Check `gradient_clip_val=1.0` is set in the Trainer; if it still diverges, lower the LR.
6. **PSNR seems crazy high (>50 dB)** — almost certainly you're computing PSNR with the wrong `data_range`, or evaluating on the *clean* image by accident.
7. **Out-of-memory on Colab** — drop the batch size, or drop the model size. `precision="16-mixed"` also halves memory for activations.
