# Contributing

## Adding an exercise

1. Create `exercises/<NN_chapter>/<NN_name>/exercise.py` and
   `solutions/<NN_chapter>/<NN_name>/exercise.py`. pytest globs the tree, so
   there is nothing to register.

2. Write the **solution first**. It is the specification, and writing it first
   is the only reliable way to find out whether the exercise is well posed.

3. Derive the starter from it: keep the tests verbatim, replace the
   implementation with a stub or a plausibly-wrong version, and mark it `TODO`.

4. Run `./scripts/check-course.sh` (or `./scripts/check-course.sh 04_` for one
   chapter while you work).

## What the checker enforces

- **Every solution passes.** A solution that does not is worse than no
  solution.
- **Every starter fails** — to import or to pass. An exercise that already
  passes teaches nothing, and this is the most common way for the course to
  rot as NumPy and PyTorch change underneath it.
- **A starter that does not import says so** in a `NOTE` line in its header
  comment, so nobody is left staring at a traceback the course did not warn
  them about. The converse is checked too: a `NOTE` that promises an import
  error fails the check if the starter in fact imports.

## What makes a good exercise here

- **The header comment is the teaching material.** There is no separate book,
  so the comment has to explain the idea, why it exists, and what goes wrong
  without it. Cite the book and section, or the docs page, where there is one.

- **The bug should be one somebody would really write.** A starter that fails
  because a function returns `0` teaches nothing. A starter that fails because
  it normalised rows with `axis=0`, or fed softmax output to
  `CrossEntropyLoss`, teaches the thing.

- **The tests are the specification.** Include at least one test that catches
  a plausible shortcut — the copy that looks like a view, the mask built with
  `and`, the eval loss computed in training mode.

- **Prefer a failure the learner can diagnose.** `np.testing.assert_allclose`
  and `torch.testing.assert_close` print both sides and the mismatch; a bare
  `assert (a == b).all()` prints `False`.

- **Tests must be deterministic and fast.** Seed everything
  (`np.random.default_rng(seed)`, `torch.manual_seed`, a `torch.Generator` for
  loaders). Performance tests compare against a reference in the same file
  with a generous ratio, never an absolute time. Everything runs on the CPU in
  a few seconds; the capstone's training step is the only thing allowed to
  take longer than five.

- **Use the current spellings.** NumPy 2.x and PyTorch 2.x only: no
  `np.random.seed`, `np.in1d`, `np.float_`; no `torch.cuda.amp`, `torch.jit`,
  `torch.cholesky`. `ruff check --select NPY201` catches the NumPy ones.

## Style

`ruff` formats and lints, configured in `pyproject.toml`, and is not up for
debate. Run `./npt format` and `./npt lint` before committing.

Exercise prose is British-flavoured plain English: short sentences, no
exclamation marks, no "simply", no "obviously".
