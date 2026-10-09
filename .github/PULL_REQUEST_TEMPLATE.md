<!--
CONTRIBUTING.md has the full policy; the lines below are the ones a reviewer checks
first. One issue, one pull request, verified both ways.
-->

## What is wrong

Closes #

## The fix

<!-- What you changed, and why this is the fix rather than a workaround. -->

## How it was verified

<!--
Name the test and say that you ran it both ways: failing before your change, passing
after it. For a text-only change, name the line of code that establishes the correct
reading instead.
-->

## What changes for someone on the current `main`

<!--
If a computed value moves, give the old value, the new value, and what produced each,
and say which columns, saved artifacts, or notebooks are affected. Write "nothing
computed changes" if that is the case.
-->

- [ ] This pull request fixes **one** defect, described in **one** issue, and closes it.
- [ ] A test fails before this change and passes after it, or the change computes nothing.
- [ ] No `.ipynb` was hand-edited: the `.py` changed and it was synced (see CONTRIBUTING.md, which gives the command for a text change and the one for a code change).
- [ ] Any change to a computed value is stated above, with both values and their source.
- [ ] This redefines no convention without an issue agreeing the intended behavior first.

## AI tools

<!--
Say whether you used them and for which parts. Using them is fine. You answer for
every line either way, and you should be able to explain it in review.
-->
