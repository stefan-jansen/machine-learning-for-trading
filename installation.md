# Installation Instructions

## How to install the required libaries

### Install miniconda

The notebooks use a virtual environment based on [miniconda3]() that you need to install first. You can find detailed instructions for various operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a virtual conda environment

[conda] is the package manager provided by the [Anaconda](https://www.anaconda.com/) python distribution that is tailored to faciliate the installation of data science libraries.

Just like there are [virtual environments](https://docs.python.org/3/tutorial/venv.html) for generic python installations, conda permits the creation of separate [environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that are based on the same interpreter (miniconda3 if you followed the above instructions) but can contain different package and versions of packages. See also [here](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307) for a more detailed tutorial.

You can create a new conda environment with name `env_name` and one or more packages with a specific version number using the command: 
```python
conda create --name env_name package=version_number 
```
e.g.
```python
conda create --name pandas_environment pandas=0.24
```
### Create conda environment from file for this book

Here, we will create an environment from a file to ensure you install the versions the code has been tested with. The environment specs are in the file `environment_[linux|mac_osx].yml` in the root of this repo, where you should choose the one corresponding to your operating system. To create the environment with the name `ml4t` (specified in the file), just run:

```python
conda env create -f environment_linux.yml
```

or 

```python
conda env create -f environment_mac_osx.yml
```
from the command line in the root directory.

#### Know Issues

In case `conda` throws a `RemoveError`, a quick fix [can be](https://github.com/conda/conda/issues/8149):

```python
conda update conda
``` 

possibly adding `--force`.

### Activate conda environment

After you've create it, you can activate the environment using its name, which in our case is `ml4t`:

```python
conda activate ml4t
```

To deactivate, simply use

```python
conda deactivate
```

## Set up jupyter extensions

jupyter notebooks can use a range of [extentsion](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) provided by the community. There are many useful ones that are described in the [documentation](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/).

The notebooks in this repo are formatted to use the [Table of Contents (2)](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html) extension. For the best experience, activate it using the Configurator inthe [Nbextensions](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator) tab available in your browser after starting the jupyter server. Modify the settings to check the option 'Leave h1 items out of ToC' if not set by default.