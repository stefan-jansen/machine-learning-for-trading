# Contributing

This repository is the maintained code base for *Machine Learning for Trading, 3rd
Edition*. Corrections are welcome and they are the point of it. Please read this before
opening an issue or a pull request: it is short, and following it is what makes a
contribution reviewable.

## What belongs here

A contribution fixes something that is wrong:

- Code that raises, hangs, or installs wrong.
- A computed result that is incorrect, including one a notebook's committed outputs
  currently show.
- A notebook whose displayed number does not equal what its own code computes.
- A broken data path, download, or environment pin.
- A statement in notebook prose that the code next to it contradicts.
- A typo in text or a label in a figure.

A verified defect gets corrected even when the correction changes a number the committed
outputs show. A committed output is not a reason to keep incorrect code. What a report
needs is the expected behavior, the evidence for it, and the compatibility impact: which
columns, artifacts, or downstream notebooks change, and whether a saved result has to be
regenerated.

Changing a convention is a different request from fixing a defect. An annualization
factor, a sign or return convention, a default, or a label definition is used by code
elsewhere in the repository, so changing one needs agreement on the intended behavior
before the code moves. Open an issue describing what the quantity should be and why, with
the derivation. That is not a refusal to fix numerical bugs; it is the step that
distinguishes a bug from a preference.

## What does not belong here

- New chapters, new notebooks, new models, new features, or new datasets. The scope of the
  repository is fixed.
- Restructuring, renaming, or reorganizing code that works.
- Style preferences. The conventions are deliberate and listed in
  [`.github/copilot-instructions.md`](.github/copilot-instructions.md): `symbol` and
  `timestamp` as the canonical schema, Polars over pandas, the registry as the only source
  of results. A pull request that changes one of these to a more common convention will be
  closed.
- Dependency bumps with no defect behind them.

## How to contribute

1. **Open an issue for one problem**, with a minimal reproduction: the commit you are on,
   the command or cell you ran, the output you got, and the output you expected. For
   anything numeric, say where the expected value comes from. An issue listing fifteen
   unrelated problems is a bundle and will be split before anything is reviewed.
2. **One pull request addressing that one issue**, and closing it. A pull request bundling
   unrelated fixes will be closed with a request to split it: one doubtful change blocks
   every good one in the same branch, and the review cost scales with the worst item.
3. **Verify the change, and say how.** Include a test that fails before your change and
   passes after it, and state in the pull request that you ran it both ways. For a prose
   fix, say which line of code establishes the correct reading. For a changed number, show
   the old value, the new value, and what produced each.
4. **Keep the number of open pull requests small** until the first ones are reviewed.

### Notebooks

Every notebook is a paired `.py` and `.ipynb`. The `.py` is the source; the `.ipynb` is
generated and its outputs are a real execution, stamped in `metadata.ml4t_provenance`.

- Edit the `.py`. Never hand-edit the `.ipynb`.
- For a **text change** - prose, a figure's alt text, adding or deleting or retagging a
  markdown cell - fold it in with
  `python .github/scripts/notebook_provenance.py sync-prose <notebook>.py`, or `sync-alt`
  if you changed alt text. These keep the existing outputs and the original execution
  stamp, and they are what the CI gate expects. Editing words inside a markdown cell
  also survives a plain `jupytext --sync`, but adding, deleting or retagging one does
  not: the stamp then describes the previous source and the gate rejects the notebook
  as a stale render even though nothing computed changed. `sync-prose` covers both, so
  use it for any text change and you will not have to tell them apart.
- For a **code change**, run `jupytext --sync`. The notebook now needs re-executing, and
  that runs here on the reference data; say so in the pull request and leave the outputs
  alone. `notebook_provenance.py clear <notebook>.ipynb` drops the outputs and the stamp
  so the notebook claims nothing, which the gate accepts.
- Do not clear or regenerate outputs to make a diff look clean, and do not commit a
  notebook you re-executed on your own machine: CI checks that a committed `.ipynb` is
  the paired `.py` executed in a real environment.

### AI tools

You may use them. You answer for what you submit, whoever or whatever wrote it.

- **Say whether you used AI tools, and for which parts.**
- **Be able to explain every line you submit**, in review, in your own words. A reply to a
  review question that does not engage with the question ends the review.
- **A plausible-looking fix is not a verified one.** Running the test both ways is what
  establishes that a change does what it claims, and it is the step that cannot be skipped.

A pull request that does not follow this may be closed without review. You are welcome to
reopen it once it does.

## Reporting without a fix

A clear issue with a reproduction is a real contribution and often the more useful one.
You do not need to send code.
