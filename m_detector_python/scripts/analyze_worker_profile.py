# analyze_worker_profile.py
import pstats
import sys
from pstats import SortKey

def analyze_profile(filename: str):
    """
    Loads a .prof file and prints the most relevant stats.
    """
    if not filename:
        print("Usage: python analyze_worker_profile.py <path_to_profile.prof>")
        return

    print(f"\n--- Analyzing Profile: {filename} ---\n")
    
    stats = pstats.Stats(filename)
    
    # strip_dirs() removes the long path prefixes, making the output cleaner
    stats.strip_dirs()

    # --- Sort by CUMULATIVE TIME ---
    # This is the most important view. It shows which high-level functions
    # are the biggest bottlenecks, including the time spent in all sub-functions.
    print("--- Top 30 Functions by Cumulative Time (Overall Bottlenecks) ---")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    
    # --- Sort by TOTAL TIME (tottime) ---
    # This view is useful for finding specific, individual functions that
    # take a lot of time, even if they don't call many other functions.
    # It helps find the "grind" functions doing the raw computation.
    print("\n" + "="*80 + "\n")
    print("--- Top 30 Functions by Total Time (Raw Computation Hotspots) ---")
    stats.sort_stats(SortKey.TIME).print_stats(30)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        profile_file = sys.argv[1]
        analyze_profile(profile_file)
    else:
        print("Usage: python analyze_worker_profile.py <path_to_profile.prof>")