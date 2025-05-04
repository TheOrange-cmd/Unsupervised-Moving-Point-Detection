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
conda create -n mdet_env python=3.9 
conda activate mdet_env
```
Install the required build, C++, and python libraries:
```
conda install cmake make gcc_linux-64 gxx_linux-64
conda install eigen pcl opencv tbb yaml-cpp
conda install libglu xorg-libx11 xorg-libxext xorg-libxfixes mesa-libgl-devel-cos7-x86_64 mesa-libgl-cos7-x86_64
conda install pybind11 libxcrypt pytest
python -m pip install jupyterlab numpy k3d matplotlib pyquaternion ipywidgets PyYAML nuscenes-devkit
```

To clean, build, test, and rerun test failures (to get only the failed output at the end in the terminal):

```
conda activate mdet_env
cd ~/Unsupervised-Moving-Point-Detection/build
make clean && cmake .. && make && ctest && ctest --rerun-failed -V
```