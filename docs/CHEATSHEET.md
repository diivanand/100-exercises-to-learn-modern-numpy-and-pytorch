# Modern NumPy and PyTorch cheatsheet

A one-page reminder of the decisions this course keeps asking you to make.
Each entry names the exercise that covers it.

## Views or copies — 01.04, 03.01, 07.04

| Operation | NumPy | PyTorch |
|---|---|---|
| basic slice `a[2:5]`, `a[::2]` | view | view |
| `reshape` / `view` | view if possible | `view` always a view (raises otherwise); `reshape` copies if needed |
| transpose `.T` / `permute` | view (strides swapped) | view, non-contiguous |
| boolean or integer-array index | copy | copy |
| arithmetic `a + b` | new array | new tensor |
| `a += b`, `t.add_(b)` | in place | in place |

Check with `np.shares_memory(a, b)` / `t.data_ptr()`; force with `.copy()` /
`.clone()`; make contiguous with `np.ascontiguousarray` / `.contiguous()`.

## Broadcasting — 02.01, 02.02, 07.07

Align shapes from the right; a dimension matches if equal or 1; a missing
dimension counts as 1. To broadcast a per-row value over columns, give it a
trailing axis: `row_scale[:, None]`. `(n,) + (n, 1)` is `(n, n)` — the classic
silent bug.

## Random numbers — 04.05, 04.06, 12.04

```python
rng = np.random.default_rng(seed)        # never np.random.seed / np.random.rand
rng.normal(size=...), rng.integers(lo, hi, endpoint=...), rng.permutation(n)
children = rng.spawn(n_workers)          # independent streams, not seed + i

torch.manual_seed(seed)                  # CPU and every device
g = torch.Generator().manual_seed(seed)  # for DataLoader(shuffle=True, generator=g)
torch.use_deterministic_algorithms(True) # when bit-for-bit matters
```

## Linear algebra — 04.02, 04.03, 04.04

| Want | Use |
|---|---|
| solve `Ax = b` | `np.linalg.solve(A, b)` / `torch.linalg.solve` — not `inv(A) @ b` |
| least squares | `np.linalg.lstsq(A, b, rcond=None)` / `torch.linalg.lstsq` |
| symmetric eigen | `np.linalg.eigh` / `torch.linalg.eigh` (sorted, real) |
| low-rank | `np.linalg.svd(A, full_matrices=False)` |
| polynomial fit | `np.polynomial.Polynomial.fit` — not `np.polyfit` / `poly1d` |

`torch.cholesky`, `torch.qr` are gone (2.14): everything lives in `torch.linalg`.

## NumPy 2 spellings — 01.09, 06.07

| Removed | Use |
|---|---|
| `np.float_`, `np.NaN`, `np.Inf` | `np.float64`, `np.nan`, `np.inf` |
| `np.in1d` | `np.isin` |
| `np.trapz` | `np.trapezoid` |
| `np.product`, `np.cumproduct`, `np.alltrue` | `np.prod`, `np.cumprod`, `np.all` |
| `np.row_stack` | `np.vstack` |
| `arr.ptp()` | `np.ptp(arr)` |
| `np.array(x, copy=False)` | `np.asarray(x)` (`copy=False` now raises if a copy is needed) |
| `np.char.*` | `np.strings.*` |

NEP 50: Python scalars are weak (`np.float32(3) + 3.0` stays float32); NumPy
scalars are not (`float32_array + np.float64(3)` becomes float64).

## Autograd — chapter 08

| Need | Use |
|---|---|
| a trainable tensor | `torch.nn.Parameter` inside a `Module`, or `requires_grad=True` |
| gradients | `loss.backward()`; they **accumulate** in `.grad` |
| reset | `optimizer.zero_grad()` (sets to `None` by default) |
| update without recording | `with torch.no_grad():` |
| evaluation | `with torch.inference_mode():` + `model.eval()` |
| a value, no graph | `.detach()` (`.item()` for a Python number) |
| second derivatives | `torch.autograd.grad(..., create_graph=True)` |
| per-sample gradients | `torch.func.vmap(torch.func.grad(f))` |
| check your backward | `torch.autograd.gradcheck` in float64 |

## Training — chapter 09

- `CrossEntropyLoss` takes **logits** and class indices; `BCEWithLogitsLoss`
  takes logits — never apply softmax/sigmoid first (09.04).
- Order per step: forward → loss → `zero_grad` → `backward` → (clip) →
  `optimizer.step()` → `scheduler.step()` (09.06, 09.08, 09.10).
- `model.train()` for training, `model.eval()` for anything else: Dropout and
  BatchNorm behave differently (09.07).
- `AdamW(weight_decay=…)` is decoupled decay; `Adam(weight_decay=…)` is L2 (09.05).
- Checkpoint model, optimiser and scheduler `state_dict`s together; load with
  `torch.load(path, weights_only=True)` (09.08).

## Data — chapter 10

`Dataset` (`__len__`, `__getitem__`) → `DataLoader(batch_size, shuffle,
generator, collate_fn, num_workers)`. Compute normalisation statistics on the
training split only (10.05). Ragged samples need a `collate_fn` (10.04).
`num_workers > 0` needs the `if __name__ == "__main__":` guard on macOS (10.07).

## Modern PyTorch — chapter 11

```python
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
compiled = torch.compile(model)                              # inductor on the GPU box
with torch.amp.autocast(device.type, dtype=torch.bfloat16):  # not torch.cuda.amp
    ...
scaler = torch.amp.GradScaler("cuda")                        # float16 only
F.scaled_dot_product_attention(q, k, v, is_causal=True)      # fused attention
torch.utils.benchmark.Timer(...)                             # syncs and warms up for you
torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
torch.export.export(model, (example,))                       # not torch.jit.trace
```

## Testing — 06.05, 12.03, 12.06

`np.testing.assert_allclose(a, b, rtol=, atol=)` and
`torch.testing.assert_close(a, b)` print the mismatch; `==` on floats does
not. Three tests every model deserves: output shape, every parameter gets a
gradient and changes after a step, and it can overfit one batch.

## The tools

```sh
./npt format --check   # ruff format
./npt lint             # ruff check, including NPY201 (NumPy 2 migration)
./npt verify           # everything
```
