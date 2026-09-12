# 100 exercises to learn modern NumPy and PyTorch

> **Status (12 September 2026).** I have not started this course yet. All 100
> exercises were drafted with AI assistance (Claude) from the books and
> documentation listed at the end of this page, and every solution passes its
> tests on my machine, but none of them has been through my own hands: an
> exercise I have not reached may contain mistakes in its prose, its starter
> or its tests, and I fix those as I get to each one. I will update this note
> as I go.

Learn NumPy 2 and PyTorch 2 by fixing, finishing and writing 100 small
programs.

Every exercise is a single self-contained file that states a problem and checks
your answer. You edit the code, run the tests, and move on when they pass.
There is no lecture to sit through and nothing to read that is not next to the
code it is about.

This is the Python counterpart to
[100 exercises to learn modern C++](https://github.com/diivanand/100_exercises_to_learn_modern_cpp),
and both follow the format of
[100 exercises to learn Rust](https://github.com/mainmatter/100-exercises-to-learn-rust).

## Requirements

- **Python 3.12 or 3.13.**
- **[uv](https://docs.astral.sh/uv/)** — `brew install uv`. It creates the
  virtual environment and installs everything on first use.

Nothing else. The course depends on NumPy 2.1+ and PyTorch 2.6+ (it was written
against NumPy 2.5 and PyTorch 2.14, the current releases in September 2026),
plus pytest and ruff for development. No GPU is needed: every test runs on the
CPU. Several exercises say what to try on a GPU if you have one.

## Getting started

```sh
git clone <this repo>
cd 100-exercises-to-learn-modern-numpy-and-pytorch
./npt next
```

`./npt next` runs the first exercise that is not yet passing and shows you its
failure. That is the whole loop:

1. Run `./npt next`.
2. Open the file it names.
3. Make the tests pass.
4. Repeat.

When every exercise passes, you are done.

## The commands

```
./npt next                  the main loop: run the first unfinished exercise
./npt test <filter>         run one exercise    (./npt test 04_02, or solve_not_inv)
./npt verify                run everything, as CI does
./npt list                  the curriculum, and where you are in it
./npt solution <filter>     diff your work against the reference solution
./npt format [--check]      ruff format the tree
./npt lint                  ruff check the tree
./npt clean                 remove caches
```

A filter is any substring of an exercise id, so `04`, `04_02`, `solve_not_inv`
and `02_solve_not_inv` all select 04.02.

## How an exercise works

Each `exercises/<chapter>/<exercise>/exercise.py` has three parts:

- **A header comment** explaining the idea, why it exists, and where it bites.
  This is the teaching material; there is no separate book. Where a claim
  comes from a book, the header says which one and which section.
- **The code you edit**, marked with `TODO`.
- **The tests**, which are the specification. Read them — several exercises
  have a test that exists specifically to catch a plausible shortcut.

A few exercises **start as an error at import time**, because the bug they are
about is one Python cannot get past (a NumPy 1.x spelling that NumPy 2 removed,
say). Those say so in a `NOTE` in their header comment.

pytest collects the `exercise.py` files directly (see `pyproject.toml`), so
`uv run pytest exercises/04_linear_algebra_and_random/02_solve_not_inv/exercise.py`
is exactly what `./npt test 04_02` runs.

## The curriculum

| # | Chapter | What it covers |
|---|---------|----------------|
| 00 | Getting started | the workflow, reading a pytest failure |
| 01 | Arrays and dtypes | creating arrays, dtypes and overflow, shape and reshape, slices are views, boolean masks, fancy indexing, axis semantics, NaN and inf, NumPy scalars and NEP 50 |
| 02 | Broadcasting and vectorisation | the broadcasting rules, adding axes, replacing loops, ufuncs and `out=`, `where`/`select`, pairwise distances, `einsum`, why `np.vectorize` is a loop |
| 03 | Views, memory and performance | views vs copies, strides, contiguity and order, in-place operations, avoiding temporaries, measuring before optimising, memory footprint |
| 04 | Linear algebra and random | `@` and `matmul`, `solve` not `inv`, least squares, `eigh` and SVD, `default_rng`, seeds and spawning, norms and normalisation |
| 05 | Structured data and I/O | sorting and searching, `unique` and set operations, stacking and splitting, `save`/`load`/`memmap`, `datetime64` and `np.strings`, the `Polynomial` API |
| 06 | NumPy idioms | `cumsum`/`diff` tricks, `sliding_window_view`, grids, histograms and binning, `np.testing`, floating-point pitfalls, migrating NumPy 1.x code to 2.x |
| 07 | Tensors | creation, the NumPy bridge, dtypes and devices, views and contiguity, in-place ops, `gather`/`scatter`, broadcasting and `einsum`, `squeeze`/`unsqueeze`/`permute` |
| 08 | Autograd | `requires_grad` and `backward`, gradient accumulation, `no_grad` and `inference_mode`, the graph and leaves, custom `autograd.Function`, `gradcheck` and higher-order gradients, `torch.func`, Jacobians |
| 09 | Modules and training | linear regression by hand, `nn.Module` and parameters, `Sequential` and init, losses take logits, optimisers and `zero_grad`, the training loop, train/eval modes, schedulers and checkpoints, init and activations, clipping and accumulation |
| 10 | Data | `Dataset`, `TensorDataset` and `random_split`, `DataLoader` batching, `collate_fn`, transforms and normalisation, samplers, workers and pinning, `IterableDataset` |
| 11 | Modern PyTorch | device-agnostic code, `torch.compile`, `autocast` and `GradScaler`, `scaled_dot_product_attention`, `torch.utils.benchmark`, `channels_last` and TF32, the profiler, `torch.export`, `functional_call` and ensembles |
| 12 | Debugging and testing | shape errors, NaN detection, `assert_close`, reproducibility, hooks, testing models |
| 13 | Capstone | a character-level transformer: tokeniser and embeddings, causal self-attention, the block, the training loop, generation |

## Solutions

Every exercise has a reference solution in `solutions/`, with comments
explaining the choices rather than just the mechanics.

```sh
./npt solution 04_02    # diff your work against it
```

Try to reach for them only after you have a failing attempt of your own — the
diff teaches more when you have already made the decision it disagrees with.

To run them all:

```sh
uv run pytest solutions -q
```

## Using this with PyCharm or IntelliJ

Open the directory. The IDE detects `.venv/` (created by `uv sync` or the
first `./npt` command) as the project interpreter and `pyproject.toml` as the
pytest configuration, so the green *run* gutter icons next to each `test_`
function do what `./npt test` does. `ruff` is picked up by the Ruff plugin.

## Where this material comes from

The header comments cite these by section, so you can read the source of any
claim:

- **Robert Johansson, *Numerical Python*, 3rd ed. (Apress, 2024)** — the
  NumPy chapters: array object, dtypes, memory order, indexing, vectorised
  expressions, linear algebra, random numbers, I/O.
- **Daniel Voigt Godoy, *Deep Learning with PyTorch Step-by-Step*** — the
  PyTorch chapters: tensors, autograd, the training loop, datasets and
  loaders, losses, modes, checkpoints, seeds.
- **Volpe et al., *Deep Learning Crash Course* (No Starch, 2026)** and
  **Ronald Kneusel, *Practical Deep Learning* (2nd ed.) and *Math for Deep
  Learning*** — backpropagation, attention and the transformer capstone.
- **Brett Slatkin, *Effective Python*, 3rd ed.** — the Python idioms, cited
  by item number.
- **Wes McKinney, *Python for Data Analysis*** — framing for the advanced
  NumPy material (its API is 2012-era; the exercises use the NumPy 2 spellings).
- The **NumPy 2.0 migration guide** and the **PyTorch 2.14 release notes**,
  so that nothing here teaches a spelling that has since been removed or
  deprecated (`np.random.seed`, `np.in1d`, `torch.cuda.amp`, `torch.jit`, …).

## License

The exercises and solutions are released under CC BY-NC 4.0, matching the Rust
course this is modelled on.
