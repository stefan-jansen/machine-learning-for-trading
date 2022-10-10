---
name: zipline installation problem
about: 
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
I created a conda environment following the installation instructions on this book’s Github repo (using ml4t-base.yml) on my Mac, but when I tried to use zipline, I encountered a few problems. 

**Problem 1. To Reproduce**
1. run `zipline run —help`
2. I got the following error:

`
Traceback (most recent call last):
  File "/Users/Chen/miniconda3/envs/ml4t/bin/zipline", line 7, in <module>
    from zipline.__main__ import main
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/__init__.py", line 29, in <module>
    from .utils.run_algo import run_algorithm
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/utils/run_algo.py", line 24, in <module>
    from zipline.pipeline.data import USEquityPricing
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/pipeline/__init__.py", line 1, in <module>
    from .classifiers import Classifier, CustomClassifier
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/pipeline/classifiers/__init__.py", line 1, in <module>
    from .classifier import (
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/pipeline/classifiers/classifier.py", line 22, in <module>
    from zipline.pipeline.term import ComputableTerm
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/pipeline/term.py", line 45, in <module>
    from .domain import Domain, GENERIC, infer_domain
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/pipeline/domain.py", line 27, in <module>
    from zipline.country import CountryCode
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/country.py", line 10, in <module>
    class CountryCode(object):
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/country.py", line 55, in CountryCode
    TURKEY = code("TURKEY")
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/country.py", line 7, in code
    return countries_by_name[name].alpha2
KeyError: 'TURKEY'
`
3. I then solved the above problem by changing country.py’s line 55 from `TURKEY=code(“TURKEY”)` to `TURKEY=code("TÜRKIYE")`.

**Problem 2. To Reproduce**
1. run `zipline ingest -b quandl`
2. I got the following error:

`
Traceback (most recent call last):
  File "/Users/Chen/miniconda3/envs/ml4t/bin/zipline", line 11, in <module>
    sys.exit(main())
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/click/core.py", line 1657, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/__main__.py", line 389, in ingest
    bundles_module.ingest(
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/data/bundles/core.py", line 415, in ingest
    daily_bar_writer.write(())
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/data/bcolz_daily_bars.py", line 209, in write
    return self._write_internal(it, assets)
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/data/bcolz_daily_bars.py", line 250, in _write_internal
    columns = {
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/zipline/data/bcolz_daily_bars.py", line 251, in <dictcomp>
    k: carray(array([], dtype=uint32_dtype))
  File "bcolz/carray_ext.pyx", line 1063, in bcolz.carray_ext.carray.__cinit__
  File "bcolz/carray_ext.pyx", line 1132, in bcolz.carray_ext.carray._create_carray
  File "/Users/Chen/miniconda3/envs/ml4t/lib/python3.8/site-packages/bcolz/utils.py", line 113, in to_ndarray
    if array.dtype != dtype.base:
AttributeError: 'NoneType' object has no attribute 'base'
`
3. I didn't solve this problem.

**Environment**
MacOS Monterey version 12.6
