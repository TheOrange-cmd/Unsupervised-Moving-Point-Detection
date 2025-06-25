Tested with cuda 12.4, linux

conda create -n mdetector_env python=3.9 -y
conda activate mdetector_env
conda install -c conda-forge numpy pandas matplotlib pyyaml scipy scikit-learn ray-all optuna hdbscan -y
conda install esri::nuscenes-devkit -y
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0%2Bcu124.html
python -m pip install nuscenes-devkit