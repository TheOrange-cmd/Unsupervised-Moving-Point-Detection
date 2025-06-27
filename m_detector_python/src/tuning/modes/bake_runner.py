# src/tuning/modes/bake_runner.py 

import ray
import optuna
import torch
import logging
import pickle 
from pathlib import Path
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from ...config_loader import MDetectorConfigAccessor
from ...core.m_detector.base import MDetector
from ...data_utils.nuscenes_helper import NuScenesProcessor
from ..shared_utils import deep_update_dict

def run(config: dict, source_study_name: str, db_dir: Path):
    logger = logging.getLogger(__name__)


    logger.info("--- Starting Bake Mode ---")
    storage_path = f"sqlite:///{db_dir / source_study_name}.db"
    
    try:
        study = optuna.load_study(study_name=source_study_name, storage=storage_path)
        best_trial = study.best_trial
        best_params = best_trial.params
        logger.info(
            f"Loaded best trial [bold cyan]#{best_trial.number}[/bold cyan] from study '[bold green]{source_study_name}[/bold green]'\n"
            f"  -> Trial Score: {best_trial.value:.4f}. Using its parameters for baking."
        )
    except Exception as e:
        logger.error(f"Could not load study '{source_study_name}'. Error: {e}")
        return

    bake_config = deep_update_dict(config.copy(), best_params)
    accessor = MDetectorConfigAccessor(config_dict=bake_config)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = MDetector(config_accessor=accessor, device=device, logger_name=__name__)
    data_actor_handle = ray.get_actor("nuscenes_data_service")
    processor = NuScenesProcessor(
        data_actor=data_actor_handle, config_accessor=accessor,
        progress_actor=None, worker_id=0, logger_name=__name__
    )

    cached_data = []
    scene_indices = accessor.get_processing_settings()['scene_indices_to_run']

    if isinstance(scene_indices, str) and scene_indices.lower() == 'all':
        total_scenes = ray.get(data_actor_handle.get_scene_count.remote())
        scene_indices = set(range(total_scenes))
    else:
        scene_indices = set(scene_indices)
    
    logger.info("Calculating total number of sweeps to process...")
    total_sweeps = 0
    for scene_idx in scene_indices:
        scene_rec = ray.get(data_actor_handle.get_scene_record.remote(scene_idx))
        sweep_tokens = ray.get(data_actor_handle.get_scene_sweep_tokens.remote(scene_rec['token']))
        total_sweeps += len(sweep_tokens)
    logger.info(f"Preparing to bake [bold yellow]{total_sweeps}[/bold yellow] sweeps across [bold yellow]{len(scene_indices)}[/bold yellow] scenes.")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn(),
    ) as progress:
        sweep_task = progress.add_task("[green]Baking sweeps...", total=total_sweeps)

        for scene_idx in scene_indices:
            frame_generator = processor.process_scene_for_baking(scene_idx, detector)
            for frame_data in frame_generator:
                cached_data.append(frame_data)
                progress.update(sweep_task, advance=1)

        if not cached_data:
            logger.error("Bake process finished, but no data was generated. Check dataset paths and detector initialization.")
            return
            
        logger.info(f"Bake processing complete. Generated data for [bold cyan]{len(cached_data)}[/bold cyan] frames.")
        logger.info("Serializing baked data to disk... (this may take a moment)")
        
        bake_id = f"{source_study_name}_trial{best_trial.number}"
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        
        output_path = cache_dir / f"{bake_id}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(cached_data, f)
        
        logger.info(f"[bold green]Bake successful![/bold green]")
        logger.info(f"Saved baked data to: [yellow]{output_path}[/yellow]")
        logger.info(f"Use the following argument for the next step: [bold cyan]--bake-id {bake_id}[/bold cyan]")