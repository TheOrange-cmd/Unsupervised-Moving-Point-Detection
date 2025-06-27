Tested with cuda 12.4, linux

conda create -n mdetector_env python=3.9 -y
conda activate mdetector_env
conda install -c conda-forge numpy pandas matplotlib pyyaml scipy scikit-learn ray-all optuna hdbscan -y
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0%2Bcu124.html
python -m pip install nuscenes-devkit


To enable scrolling in tmux:
tmux set -g mouse on

Optuna setup:
(optional):
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.ray_scripts.run_experiment \
    --mode tune-full \
    --study-name geo_study_v1 \
    --n-trials 500

python -m src.ray_scripts.run_experiment \
    --mode bake \
    --study-name "Baking geo_study_v1" \
    --source-study-name geo_study_v1

python -m src.ray_scripts.run_experiment \
    --mode tune-refinement \
    --study-name refinement_on_geo_v1 \
    --n-trials 1000 \
    --bake-id geo_study_v1_trial388