#!/bin/bash

wget https://github.com/git-lfs/git-lfs/releases/download/v2.5.2/git-lfs-linux-amd64-v2.5.2.tar.gz

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

source ~/.bashrc

sudo apt-get install -y zlib1g-dev libbz2-dev libreadline6-dev libsqlite3-dev g++ make libssl-dev libsm-dev libxrender1 unzip
pyenv install 3.6.5
pip install --upgrade pip
pip install -r requirements.txt

wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/1/cuda-repo-ubuntu1604-9-0-local-cublas-performance-update_1.0-1_amd64-deb
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/2/cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/3/cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-3_1.0-1_amd64-deb
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/patches/4/cuda-repo-ubuntu1604-9-0-176-local-patch-4_1.0-1_amd64-deb

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local-cublas-performance-update_1.0-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local-cublas-performance-update/7fa2af80.pub

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-2_1.0-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local-cublas-performance-update-2/7fa2af80.pub

sudo dpkg -i cuda-repo-ubuntu1604-9-0-local-cublas-performance-update-3_1.0-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-local-cublas-performance-update-3/7fa2af80.pub

sudo dpkg -i cuda-repo-ubuntu1604-9-0-176-local-patch-4_1.0-1_amd64-deb
sudo apt-key add /var/cuda-repo-9-0-176-local-patch-4/7fa2af80.pub

