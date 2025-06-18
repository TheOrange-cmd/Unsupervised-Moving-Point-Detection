# scripts/profile_script.py

import cProfile
import pstats
import io
from pstats import SortKey
import os
import sys
import logging
import multiprocessing 

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    print(f"Added project root to sys.path: {PROJECT_ROOT}")

# Import main function AFTER setting up the path
# from scripts.run_mdetector_and_save import main
from scripts.run_ray_experiment import main as ray_main

def profile_and_run():
    """
    A wrapper function to be profiled.
    """
    # Configure logging 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Profile the main function
    profile_filename = "mdetector_profile_torch.prof"
    cProfile.run('ray_main()', profile_filename)

    # Print the stats
    s = io.StringIO()
    ps = pstats.Stats(profile_filename, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)
    print("\n--- Profiling Summary (Top 20 by Cumulative Time) ---")
    print(s.getvalue())

    # Save detailed stats to a file
    with open("profile_detailed.txt", "w") as f:
        ps_file = pstats.Stats(profile_filename, stream=f).sort_stats(SortKey.CUMULATIVE)
        ps_file.print_stats()
    print("Detailed profiling stats saved to profile_detailed.txt")


if __name__ == '__main__':
    profile_and_run()