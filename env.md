To create the env:

install miniconda:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts - say 'yes' to initializing Conda
source ~/.bashrc # Or open a new terminal
```
Configure channels:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Create env:
```
conda create -n mdet_env python=3.9 # Include python if needed for scripts later
conda activate mdet_env
```
Install a bunch of libs:
```
conda install cmake make gcc_linux-64 gxx_linux-64
conda install eigen pcl opencv tbb yaml-cpp
conda install libglu xorg-libx11 xorg-libxext xorg-libxfixes mesa-libgl-devel-cos7-x86_64 mesa-libgl-cos7-x86_64
```

To clean, build, and test in one go, outputting test failures:

```
conda activate mdet_env
cd ~/m_detector/standalone/build
rm -rf * && cmake .. && make && ctest --output-on-failure
```