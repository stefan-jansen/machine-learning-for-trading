# Installation instructions

This book uses (mostly) Python 3.7 and numerous libraries that require installation. The first section covers how use Docker to pull an image that contains the necessary software and create a local container that allows you to run the notebooks. The second section describes how to install the packages locally using the [Anaconda](https://www.anaconda.com/) distribution. If you are experienced and work a UNIX-based system, you can also create your own virtual environments and install the libraries required for different notebooks as needed. 

Then, we address how to work with [Jupyter](https://jupyter.org/) notebooks to view and execute the code examples. Finally, we list additional installation instructions for libraries that require non-python dependencies and discuss how to 

## Run the code using a Docker images

1. Install [Docker Desktop](https://docs.docker.com/desktop/) for [Windows](https://docs.docker.com/docker-for-windows/install/) or [Mac OS](https://docs.docker.com/docker-for-mac/install/).
    - Review Getting Started guides for [Windows](https://docs.docker.com/docker-for-windows/) or [Mac OS](https://docs.docker.com/docker-for-mac/). Under Preferences, look for Resources to find out how you can increase the memory allocated to the container; the default setting is too low given the size of the data. Increase to at least 4GB.
2. Clone the starter repo using the following command: `git clone ` and change into the new directory.
3. [Register](https://www.quandl.com/sign-up) for a (free) personal Quandl account to obtain an API key that you'll need in the next step.  
3. We'll be using an image based on the Ubuntu 20.04 OS with the Anaconda Python distribution installed. It also contains two conda environments, one to run Zipline and one for everything else. The following command does several things at once:
    ```docker
    docker run -it -v $(pwd):/home/packt/ml4t -p 8888:8888 -e QUANDL_API_KEY=<your API key> --name ml4t appliedai/packt:latest bash
    ```
    - it pulls the image from the Docker Hub account `appliedai` and the repository `packt` with the tag `latest`
    - creates a local container with the name `ml4t` and runs it in interactive mode, forwarding the port 8888 used by the `jupyter` server
    - mounts the current directory containing the starter project files as a volume in the directory `/home/packt/ml4t` inside the container
    - sets the environment variable `QUANDL_API_KEY` with the value of your key (that you need to fill in for `<your API key>`), and
    - starts a `bash` terminal inside the container, resulting in a new command prompt for the user `packt`.
4. Now you are running a shell inside the container and can access the `conda environments`.
    - Run `conda env list` to see that there are a `base`, `ml4t` (default), and a `ml4t-`zipline` environment.
    - You can switch to another environment using `conda activate <env_name>`. 
        - However, before doing so the first time, you may get an error message suggesting you run `conda init bash`. After doing so, reload the shell with the command `source .bashrc`. 
5. To run Zipline backtests, we need to `ingest` data. See the [Beginner Tutorial](https://www.zipline.io/beginner-tutorial.html) for more information. The image has been configured to store the data in a `.zipline` directory in the directory wehre you started the container (which should be the root folder of the project starter code). 
    - From the command prompt of the container shell, run
    ```bash
    conda activate ml4t-zipline
    zipline ingest
    ``` 
   - You should see numerous messages as Zipline processes around 3,000 stock price series
5. Now we would like to test Zipline and the [juypter](https://jupyter.org/) setup. You can run notebooks using either the traditional or the more recent Jupyter Lab interface; both are available in all `conda` environments. Moreover, you start jupyter from the `base` environment and switch the environment from the notebook due to the `nb_conda_kernels` package (see [docs](https://github.com/Anaconda-Platform/nb_conda_kernels)). To get started, run one of the following two commands:
    ```bash
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
    jupyter lab --ip 0.0.0.0 --no-browser --allow-root
   ```
    - There are also `alias` shortcuts for each so you don't have to type them: `nb` for the `jupyter notebook` version, and `lab` for the `jupyter lab` version.
    - The terminal will display a few messages and at the end indicate what to paste into your browser to access the jupyter server from the current working directory.
6. You can modify any of the environments using the standard conda workflow outlined below; see Docker docs for how to persist containers after making changes.

## How to install the required libraries using `conda` environments

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
| 1, 2 & 3  | 2-16     | ml4t.yml      | environments/{linux\|macos\|windows}/ml4t.yml      |
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