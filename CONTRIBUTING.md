# Contributing

This repository is the companion code for *Machine Learning for Trading, 3rd Edition*
(MIT Press, July 2026). The book is printed and is not being revised. That shapes what a
contribution can be here, so please read this before opening a pull request.

The code **is** maintained. Corrections are welcome and they are the point of this
repository.

## What belongs here

A contribution fixes something a reader hits:

- Code that raises, hangs, or installs wrong.
- A notebook whose displayed number does not equal what its own code computes.
- A broken data path, download, or environment pin.
- A statement in notebook prose that the code next to it contradicts.
- A typo in text or a label in a figure.

## What does not

- New chapters, new notebooks, new models, new features, or new datasets. The table of
  contents is printed; nothing can be added to it.
- Restructuring, renaming, or reorganizing anything that works.
- Style preferences. The repository's conventions are deliberate and listed in
  [`.github/copilot-instructions.md`](.github/copilot-instructions.md): `symbol` and
  `timestamp` as the canonical schema, Polars over pandas, no em dashes, the registry as
  the only source of results. A pull request that changes one of these to a more common
  convention will be closed.
- Dependency bumps with no defect behind them.

**Changing a published number is not a bug fix.** An annualization factor, a sign or
return convention, a default, a label definition: the printed book quotes results computed
with these, so changing one makes the book and the code disagree. If you believe one is
wrong, open an issue with the derivation. The outcome may be an erratum rather than a code
change, and that decision is the author's.

## How to contribute

1. **Open an issue first**, with a minimal reproduction: the commit you are on, the
   command or cell you ran, the output you got, and the output you expected. For anything
   numeric, say where the expected value comes from.
2. **One defect per pull request**, closing that issue. A pull request bundling unrelated
   fixes will be closed with a request to split it, because one doubtful change blocks
   every good one in the same branch and the review cost scales with the worst item.
3. **Include a test that fails on `main` and passes with your change.** Say in the pull
   request that you ran it both ways. For a notebook prose fix, instead say which line of
   code establishes the correct reading.
4. **Keep the number of open pull requests small** until the first ones are reviewed.

### Notebooks

Every notebook is a paired `.py` and `.ipynb`. The `.py` is the source; the `.ipynb` is
generated and its outputs are a real execution, stamped in `metadata.ml4t_provenance`.

- Edit the `.py`, then run `jupytext --sync`. Never hand-edit the `.ipynb`.
- Do not clear, regenerate, or re-execute outputs to make a diff look clean. CI checks
  that a committed `.ipynb` matches the `.py` it was executed from, and re-running a
  notebook on your machine replaces a published result with yours.
- A text-only change keeps the existing outputs. If your change makes the notebook compute
  something different, say so in the pull request: it needs a re-execution the maintainer
  runs.

### AI tools

You may use them. You answer for what you submit, whoever or whatever wrote it.

- **Say whether you used AI tools, and for which parts.**
- **Be able to explain every line you submit**, in review, in your own words. A generated
  reply to a review question that does not engage with the question wastes the reviewer's
  time and will end the review.
- **A plausible-looking fix is not a verified one.** The checks above exist because
  generated patches are cheap to produce and expensive to verify. Running the test both
  ways is what moves that cost back to you, and it is the one step that cannot be skipped.

A pull request that does not follow this may be closed without review. You are welcome to
reopen it once it does.

## Reporting without a fix

A clear issue with a reproduction is a real contribution and often the more useful one.
You do not need to send code.
