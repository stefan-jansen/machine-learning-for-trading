# Installation instructions

This book uses (mostly) Python 3.7 and various ML- and trading-related libraries available in three different [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) based on the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution. I developed the content on Ubuntu 20.04 while also testing on Mac OS 10.15 (Catalina). 

Depending on your OS, you may have several options to create these environments. These are, in increasing order of complexity:
 1. **Recommended**: use [Docker](https://www.docker.com/) Desktop to pull an image from [Docker Hub](https://www.docker.com/products/docker-hub) and create a local container with the requisite software to run the notebooks. 
 2. Create the [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using the provided `.yml` environment files as outlined below. However, the backtesting environment that relies on the [patched](https://github.com/stefan-jansen/zipline) version of `zipline` only exists for **Ubuntu** due to numerous version conflicts. 
 3. If you are experienced (and work on a UNIX-based system), you can also create your own virtual environments and install the libraries required for the different notebooks using `pip` as needed. 

We'll describe the first two options in turn. Then, we address how to work with [Jupyter](https://jupyter.org/) notebooks to view and execute the code examples. Finally, we list additional installation instructions for libraries that require non-python dependencies like [TA-Lib](https://mrjbq7.github.io/ta-lib/) for technical analysis.

## Running the notebooks using a Docker container

Docker Desktop is a very popular application for MacOS and Windows machines because is permits for the easy sharing of containerized applications across different OS. For this book, we have a Docker image that let's you instantiate a container to run Ubuntu 20.04 as a guest OS with the pre-installed [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on Windows 10 or Mac OS X without worrying about dependencies on your host.

### Installing Docker Desktop 

As usual, installation differs for Mac OS X and and Window 10, and requires an additional step for Windows 10 Home to enable virtualization. 

We'll cover installation for each OS separately and then address some setting adjustments necessary in both cases.

#### Docker Desktop on Mac OS X

Installing Docker Desktop on Mac OS X is very straightforward:
1. Follow the detailed guide in Docker [docs](https://docs.docker.com/docker-for-mac/install/) to download and install Docker Desktop from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac/). It also covers how Docker Desktop and Docker Toolbox [can coexist](https://docs.docker.com/docker-for-mac/docker-toolbox/).
2. Use [homebrew](https://brew.sh/) by following the tutorial [here](https://aspetraining.com/resources/blog/docker-on-mac-homebrew-a-step-by-step-tutorial).

Open terminal and run the following test to check that Docker works:
```Docker
docker run hello-world
```

Review the [Getting Started](https://docs.docker.com/docker-for-mac/) guide for Mac OS to familiarize yourself with key settings and commands.

#### Docker Desktop on Windows

Docker Desktop works on both Windows 10 Home and Pro editions; the Home edition requires the additional step of enabling the Virtual Machine Platform.  

##### Windows 10 Home: enabling the Virtual Machine Platform

You can now install Docker Desktop on Windows Home machines using the [Windows Subsystem for Linux](https://fossbytes.com/what-is-windows-subsystem-for-linux-wsl/) (WSL 2)  backend. Docker Desktop on Windows Home is a full version of Docker Desktop for Linux container development.

Windows 10 Home machines must meet certain [requirements](https://docs.docker.com/docker-for-windows/install-windows-home/#system-requirements). These include Windows 10 Home version 2004 (released May 2020) or higher. The Docker Desktop Edge release also supports Windows 10, version 1903 or higher.

Enable WSL 2 as described [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10), taking the following steps:

1. Enable the optional Windows Subsystem for Linux feature. Open PowerShell as Administrator and run:
    ```bash
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
2. Check that your system meets the requirements outlined [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10#requirements) and update your Windows 10 version if necessary.
3. Enable the Virtual Machine Platform optional feature by opening PowerShell as and Administrator and run:
    ```bash
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    ```
4. Restart your machine to complete the WSL install and update to WSL 2.
5. Download and run the Linux kernel [update package](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi). You will be prompted for elevated permissions, select ‘yes’ to approve this installation.
6. Set WSL 2 as your default version when installing a new Linux distribution by open PowerShell as Administrator and run the following command:
    ```bash
    wsl --set-default-version 2
    ```
  
##### Windows 10: Docker Desktop installation 

Once we have enabled WSL 2 for Windows Home, the remaining steps to install Docker Desktop are the same for Windows 10 [Home](https://docs.docker.com/docker-for-windows/install-windows-home/) and [Pro, Enterprise or Education](https://docs.docker.com/docker-for-windows/install-windows-home/). Refer to the linked guides for each OS version for system requirements.

1. Download and run (double-click) the installer from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-windows/).
2. When prompted, ensure the Enable Hyper-V Windows Features option is selected on the Configuration page.
3. Follow the instructions on the installation wizard to authorize the installer and proceed with the install.
4. When the installation is successful, click Close to complete the installation process.
5. If your admin account is different to your user account, you must add the user to the docker-users group. Run Computer Management as an administrator and navigate to Local Users and Groups > Groups > docker-users. Right-click to add the user to the group. Log out and log back in for the changes to take effect.

Open Powershell and run the following test to check that Docker works:
```Docker
docker run hello-world
```

Review the [Getting Started](https://docs.docker.com/docker-for-windows/) guide for Windows to familiarize yourself with key settings and commands.

### Docker Desktop Settings: memory and file sharing 

The getting started guides for each OS referenced above describe the Docker Desktop settings.

#### Increasing memory 
- Under Preferences, look for Resources to find out how you can increase the memory allocated to the container; the default setting is too low given the size of the data. Increase to at least 4GB, better 8GB or more.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
- Several examples are quite memory-intensive, for example the NASDAQ tick data and the SEC filings example in Chapter 2, and will require significantly higher memory allocation.

#### Troubleshooting file sharing permissions

We will download the code examples and data to the local drive on your host OS but run it from the Docker container by mounting your local drive as a volume. This should work fine with the current versions but in case you receive **permission errors** , please refer to the **File Sharing** sections in the Docker user guides. The Docker GUIs let you assign permissions explicitly. See also (slightly outdated) explanation [here](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c).
  
### Sourcing the code samples

You can work with the code samples by downloading a compressed version of the [GitHub repository](https://github.com/stefan-jansen/machine-learning-for-trading), or by [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) its content. The latter will result in a larger download because it includes the commit history. 

Alternatively, you can create a [fork](https://guides.github.com/activities/forking/) of the repo and continue to develop from there after cloning its content.

To work with the code locally, do the following:
1. Select a file system location where you would like to store the code and the data.
2. Using the `ssh` or `https` links or the download option provided by the green `Code` button on the [GitHub repository](https://github.com/stefan-jansen/machine-learning-for-trading), either clone or unzip the code to the target folder.
    - To clone the starter repo, run `git clone https://github.com/stefan-jansen/machine-learning-for-trading.git` and change into the new directory.
    - If you cloned the repo and did not rename it, the root directory will be called `machine-learning-for-trading`, the ZIP the version will unzip to `machine-learning-for-trading-master`.

### Get a QUANDL API Key

To download US equity data that we'll be using for several examples throughout the book in the next step, [register](https://www.quandl.com/sign-up) for a personal Quandl account to obtain an API key. It will be displayed on your [profile](https://www.quandl.com/account/profile) page.

If you are on a UNIX-based system like Mac OSX, you may want to store the API key in an environment variable such as QUANDL_API_KEY, e.g. by adding `export QUANDL_API_KEY=<your_key>` to your `.bash_profile`.  

### Downloading the Docker image and running the container

We'll be using a Docker [image](https://hub.docker.com/repository/docker/appliedai/packt) based on the Ubuntu 20.04 OS with [Anaconda](https://www.anaconda.com/)'s [miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution installed. It comes with three conda environments describe below. 

With a single Docker command, we can accomplish several things at once (see the Getting Started guides linked above for more detail):
- only on the first run: pull the Docker image from the Docker Hub account `appliedai` and the repository `packt` with the tag `latest` 
- creates a local container with the name `ml4t` and runs it in interactive mode, forwarding the port 8888 used by the `jupyter` server
- mount the current directory containing the starter project files as a volume in the directory `/home/packt/ml4t` inside the container
- set the environment variable `QUANDL_API_KEY` with the value of your key (that you need to fill in for `<your API key>`), and
- start a `bash` terminal inside the container, resulting in a new command prompt for the user `packt`.

1. Open a Terminal or a Powershell window,
2. Navigate to the directory containing the [ML4T](https://github.com/stefan-jansen/machine-learning-for-trading) code samples that you sourced above,
3. In the root directory of the local version of the repo, run the following command, taking into account the different path formats required by Mac and Windows:
    - **Mac OS**: you can use the `pwd` command as a shell variable that contains the absolute path to the present working directory (and you could use `$QUANDL_API_KEY` if you created such an environment variable in the previous step):  
        ```docker
        docker run -it -v $(pwd):/home/packt/ml4t -p 8888:8888 -e QUANDL_API_KEY=<your API key> --name ml4t appliedai/packt:latest bash
        ```
   - **Windows**: enter the absolute path to the current directory **with forward slashes**, e.g. `C:/Users/stefan/Documents/machine-learning-for-trading` instead of `C:\Users\stefan\Documents\machine-learning-for-trading`, so that the command becomes (for this example):                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                  
     ```docker
     docker run -it -v C:/Users/stefan/Documents/machine-learning-for-trading:/home/packt/ml4t -p 8888:8888 -e QUANDL_API_KEY=<your API key> --name ml4t appliedai/packt:latest bash
     ```              
4. Run `exit` from the container shell to exit and stop the container. 
5. To resume working, you can run `docker start -a -i ml4t` from Mac OS terminal or Windows Powershell in the root directory to restart the container and attach it to the host shell in interactive mode (see Docker docs for more detail).                                                                                                                                                                                                                                                                                                                                                                                                  

### Running the notebooks from the container

Now you are running a shell inside the container and can access the various [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Run `conda env list` to see that there are a `base`, `ml4t` (default), `ml4t-dl` and an `ml4t-zipline` environments that we will use as follows:

| Part(s)   | Chapters        | Name         |
|-----------|-----------------|--------------|
| 1, 2 & 3  | 2-16, Appendix  | ml4t   |
| 4         | 17-22*           | ml4t-dl  |
| all | throughout              | ml4t-zipline

> the Deep Reinforcement Learning examples require TensorFlow 2.2, which currently is only available for Linux via `conda` for GPU; the notebooks contain instructions for upgrading via `pip`. Check [here](https://anaconda.org/anaconda/tensorflow) for current CPU and [here](https://anaconda.org/anaconda/tensorflow-gpu) for current GPU version support.

- You can switch to another environment using `conda activate <env_name>`.
- Alternatively, you can switch from one environment to another from the jupyter notebook or jupyter lab thanks to the [nb_conda_kernels](https://github.com/Anaconda-Platform/nb_conda_kernels) extension (see below).
- You may see an error message suggesting you run `conda init bash`. After doing so, reload the shell with the command `source .bashrc`.

### Ingesting Zipline data

To run Zipline backtests, we need to `ingest` data. See the [Beginner Tutorial](https://www.zipline.io/beginner-tutorial.html) for more information. 

The image has been configured to store the data in a `.zipline` directory in the directory where you started the container (which should be the root folder of the starter code you've downloaded above). 

From the command prompt of the container shell, run
```bash
conda activate ml4t-zipline
zipline ingest
``` 
You should see numerous messages as Zipline processes around 3,000 stock price series.

> When running a backtest, you will likely encounter an [error](https://github.com/quantopian/zipline/issues/2517) because the current Zipline version requires a country code entry in the exchanges table of the `assets-7.sqlite` database where it stores the asset metadata.
> The linked [GitHub issue](https://github.com/quantopian/zipline/issues/2517) describes how to address this by opening the [SQLite database](https://sqlitebrowser.org/dl/) and entering `US` in the `country_code` field of the exchanges.

### Working with notebooks int the Docker container

You can run [juypter](https://jupyter.org/) notebooks using either the traditional [notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) or the more recent [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/) interface; both are available in all `conda` environments. Moreover, you start jupyter from the `base` environment and switch the environment from the notebook due to the `nb_conda_kernels` package (see [docs](https://github.com/Anaconda-Platform/nb_conda_kernels). 

To get started, run one of the following two commands:
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```
There are also `alias` shortcuts for each so you don't have to type them: 
- `nb` for the `jupyter notebook` version, and 
- `lab` for the `jupyter lab` version.

The container terminal will display a few messages while spinning up the jupyter server. When complete, it will display a URL that you should paste into your browser to access the jupyter server from the current working directory.

You can modify any of the environments using the standard conda workflow outlined below; see Docker [docs](https://docs.docker.com/storage/) for how to persist containers after making changes.  

## How to install the required libraries using `conda` environments

The code examples have been developed using Anaconda's [miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution to facilitate dependency management, in particular on Windows machines. 

If you are experienced (and work in a Unix-based environment), feel free to create your own environment using `pip`. You could just install the packages required for the notebooks you are interested in; however, challenges to get the patched Zipline version to work properly will likely remain.

### Install miniconda

The notebooks rely on three different virtual environments based on [miniconda3](https://docs.conda.io/en/latest/miniconda.html) that you need to install first. 

You can find detailed instructions for various operating systems [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create conda environment from a file for this book

[conda] is the package manager provided by the [Anaconda](https://www.anaconda.com/) python distribution that is tailored to facilitating the installation of data science libraries.

Just like for [virtual environments](https://docs.python.org/3/tutorial/venv.html) for generic python installations, `conda` permits the creation of separate [environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) that are based on the same interpreter (miniconda3 if you followed the above instructions) but can contain different package and versions of packages. See also [here](https://towardsdatascience.com/getting-started-with-python-environments-using-conda-32e9f2779307) for a more detailed tutorial.

You can create a new conda environment with name `env_name` and one or more packages with a specific version number using the command: 

```bash
conda create --name env_name package=version_number 
```
e.g.
```bash
conda create --name pandas_environment pandas=1.05
```

Here, we will create environments from files to ensure you install the library versions that the code has been tested with. There are separate environment specs for parts 1, 2, 3 and 4 as described in the following table. They differ by operating system and can be found under the respective path.

| Name          |Part(s)    | Chapters        | Path |
|---------------|-----------|-----------------|----------------------------------------------|
| ml4t          |1, 2 & 3   | 2-16, appendix  |  installation/{linux &#x7c; macos &#x7c; windows}/ml4t.yml    |
| ml4t-dl       |4          | 17-22           | installation//{linux &#x7c; macos &#x7c; windows}/ml4t_dl.yml   |
| ml4t-zipline  | all | throughout           |  installation/linux/ml4t_zipline.yml   |

> *the Deep Reinforcement Learning examples require TensorFlow 2.2, which currently is only available via `conda` for GPUC; the notebooks contain instructions for upgrading via `pip`. 

To create the environment with the name `ml4t` (specified in the file) for `linux`, from the repository's root directory, just run:

```bash
conda env create -f installation/linux/ml4t.yml
```

or, for Max OS X

```bash
conda env create -f installation/macos/ml4t.yml
```
from the command line in this directory.

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

The notebooks in this repo are formatted to use the [Table of Contents (2)](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/toc2/README.html) extension. For the best experience, activate it using the Configurator in the [Nbextensions](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator) tab available in your browser after starting the jupyter server. Modify the settings to check the option 'Leave h1 items out of ToC' if not set by default.

### Converting jupyter notebooks to python scripts

The book uses [jupyter](https://jupyter.org/) notebooks to present the code with extensive commentary and context information and facilitate the visualization of results in one place. Some of the code examples are longer and make more sense to run as `python` scripts; you can convert a notebook to a script by running the following on the command line:

```bash
$ jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```

## Additional installation instructions

### TA-Lib

For the python wrapper around TA-Lib, please follow installation instructions [here](https://mrjbq7.github.io/ta-lib/install.html).