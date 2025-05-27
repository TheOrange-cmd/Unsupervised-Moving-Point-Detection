# profile_script.py
import cProfile
import pstats
import io
from pstats import SortKey
import os
import sys
import logging
import yaml

# Add project root to sys.path to allow importing from src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    # Use tqdm.write instead of print to avoid interfering with progress bar
    print(f"Added project root to sys.path: {PROJECT_ROOT}")

# Import your main function
from scripts.run_mdetector_and_save import main

logging.basicConfig(
    level=logging.INFO,  # Capture DEBUG messages and above
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    # Optional: To write to a file as well as console:
    # filename='mdetector_debug.log',
    # filemode='w' # 'w' for overwrite, 'a' for append
)


# Profile the main function
profile_filename = "mdetector_profile.prof"
cProfile.run('main()', profile_filename)

# Print the stats
s = io.StringIO()
ps = pstats.Stats(profile_filename, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(20)  # Print top 20 functions by cumulative time
print(s.getvalue())

# Optionally, save detailed stats to a file
with open("profile_detailed.txt", "w") as f:
    ps = pstats.Stats(profile_filename, stream=f).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()