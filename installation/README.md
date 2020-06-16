# Installation instructions

# docker run -it -v $(pwd):/home/packt --name ml4t packt bash
# jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root

This book uses Python 3.7 and numerous libraries that require installation. The first section covers how to handle this using the [Anaconda](https://www.anaconda.com/) distribution. Then, we address how to work with [Jupyter](https://jupyter.org/) notebooks to view and execute the code examples. Finally, we list additional installation instructions for libraries that require non-python dependencies.

## How to install the required libaries

The book has been developed using Anaconda's miniconda distribution to facilitate dependency management, in particular on Windows machines. If you are experienced and/or work in a Unix-based environment, feel free to create your own environment using `pip`; the book uses the latest compatible versions as of May 2020 as listed in the various environment files.

### Install miniconda

The notebooks have been tested using several virtual environments based on [miniconda3](https://docs.conda.io/en/latest/miniconda.html) that you need to install first. You can find detailed instructions for various operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a virtual conda environment

[conda] is the package manager provided by the [Anaconda](https://www.anaconda.com/) python distribution that is tailored to faciliate the installation of data science libraries.

Just like there are [virtual environments](https://docs.python.org/3/tutorial/venv.html) for generic python installations, conda permits the creation of separate [environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that are based on the same interpreter (miniconda3 if you followed the above instructions) but can contain different package and versions of packages. See also [here](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307) for a more detailed tutorial.

You can create a new conda environment with name `env_name` and one or more packages with a specific version number using the command: 
```bash
conda create --name env_name package=version_number 
```
e.g.
```bash
conda create --name pandas_environment pandas=0.24
```
### Create conda environment from a file for this book

Here, we will create an environment from a file to ensure you install the versions the code has been tested with. There are separate environment specs for parts 1&2, 3 and 4 as described in the following table. They differ by operating system and can be found under the respective path.

| Part(s) | Chapters | File          | Path                                               |
|---------|----------|---------------|----------------------------------------------------|
| 1 & 2   | 2-13     | ml4t.yml      | environments/{linux\|macos\|windows}/ml4t.yml      |
| 3       | 14-16    | ml4t_text.yml | environments/{linux\|macos\|windows}/ml4t_text.yml |
| 4       | 17-22    | ml4t_dl.yml   | environments/{linux\|macos\|windows}/ml4t_dl.yml   |

To create the environment with the name `ml4t` (specified in the file) for `linux`, just run:

```bash
conda env create -f environments/linux/ml4t.yml
```

or, for Max OS X

```bash
conda env create -f environments/macos/ml4t.yml
```
from the command line in this directory.

### Separate environment for using Zipline

- The current Zipline release 1.3 has a few shortcomings such as the [dependency on benchmark data from the IEX exchange](https://github.com/quantopian/zipline/issues/2480) and limitations for importing features beyond the basic OHLCV data points.
- To enable the use of `Zipline`, I've provided a [patched version](https://github.com/stefan-jansen/zipline) that works for the purposes of this book.
    - Create a virtual environment based on Python 3.5, for instance using [pyenv](https://github.com/pyenv/pyenv)
    - After activating the virtual environment, run `pip install -U pip Cython`
    - Install the patched `zipline` version by cloning the repo, `cd` into the packages' root folder and run `pip install -e`
    - Run `pip install jupyter pyfolio`

#### Known Issues

In case `conda` throws a `RemoveError`, a quick fix [can be](https://github.com/conda/conda/issues/8149):

```bash
conda update conda
``` 

possibly adding `--force`.

### Activate conda environment

After you've create it, you can activate the environment using its name, which in our case is `ml4t`:

```bash
conda activate ml4t
```

To deactivate, simply use

```bash
conda deactivate
```
## Working with Jupyter notebooks

This section covers how to set up notebook extension that facilitate working in this environment and how to convert notebooks to python script if preferred. 

### Set up jupyter extensions

jupyter notebooks can use a range of [extentsion](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) provided by the community. There are many useful ones that are described in the [documentation](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/).

The notebooks in this repo are formatted to use the [Table of Contents (2)](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html) extension. For the best experience, activate it using the Configurator inthe [Nbextensions](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator) tab available in your browser after starting the jupyter server. Modify the settings to check the option 'Leave h1 items out of ToC' if not set by default.

### Converting jupyter notebooks to python scripts

The book uses [jupyter](https://jupyter.org/) notebooks to present the code with extensive commentary and context information and facilitate the visualization of results in one place. Some of the code examples are longer and make more sense to run as `python` scripts; you can convert a notebook to a script by running the following on the command line:

```bash
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```

For testing purposes, most directories already include notebooks converted to python scripts that are more sparsely commented.  

## Additional installation instructions

### TA-Lib

For the python wrapper around TA-Lib, please follow installation instructions [here](https://mrjbq7.github.io/ta-lib/install.html).