#!/usr/bin/env bash


sudo apt-get install software-properties-common;
sudo apt-add-repository universe;
sudo apt-get update && sudo apt upgrade -y;

sudo apt install -y zlib1g-dev build-essential libncursesw5-dev libreadline-gplv2-dev libssl-dev libgdbm-dev libc6-dev libsqlite3-dev tk-dev libbz2-dev libboost-all-dev cmake glances


sudo apt install -y ubuntu-drivers-common;
ubuntu-drivers devices;
sudo ubuntu-drivers autoinstall;
sudo reboot;

curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
pyenv install 3.6.6
pyenv global 3.6.6

pip install pandas scikit-learn scipy tables pip -U;

wget -O cuda_9.1.85_387.26_linux.run -c https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux
sudo sh cuda_9.1.85_387.26_linux.run --override
# don't accept driver install

# patches
wget -O cuda_9.1.85.1_linux.run -c https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/1/cuda_9.1.85.1_linux
sudo sh cuda_9.1.85.1_linux.run
wget -O cuda_9.1.85.2_linux.run -c https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/2/cuda_9.1.85.2_linux
sudo sh cuda_9.1.85.2_linux.run
wget -O cuda_9.1.85.3_linux.run -c https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda_9.1.85.3_linux
sudo sh cuda_9.1.85.3_linux.run

# test:
nvidia-smi
nvcc --version

#nvcc: NVIDIA (R) Cuda compiler driver
#Copyright (c) 2005-2017 NVIDIA Corporation
#Built on Fri_Nov__3_21:07:56_CDT_2017

sudo echo "/usr/local/cuda-9.1/lib64" |sudo tee -a /etc/ld.so.conf

git clone --recursive https://github.com/Microsoft/LightGBM ; cd LightGBM;
mkdir build ; cd build;
cmake -DUSE_GPU=1 .. -DOpenCL_LIBRARY=/usr/local/cuda-9.1/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-9.1/include/ ..
make -j8

python setup.py install --gpu --opencl-include-dir=/usr/local/cuda-9.1/include/ --opencl-library=/usr/local/cuda-9.1/lib64/libOpenCL.so

wget -O cuda_10.0.130_410.48_linux.run -c https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
sudo sh cuda_10.0.130_410.48_linux.run

