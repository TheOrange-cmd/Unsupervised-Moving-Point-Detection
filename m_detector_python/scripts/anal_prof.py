import pstats
from pstats import SortKey

def analyze_profile(profile_path="mdetector_profile.prof", num_stats_to_show=30):
    """
    Analyzes a .prof file and prints profiling statistics.

    Args:
        profile_path (str): Path to the .prof file.
        num_stats_to_show (int): Number of top statistics to display.
    """
    try:
        stats = pstats.Stats("/home/drugge/Unsupervised-Moving-Point-Detection/m_detector_python/mdetector_profile.prof")
    except FileNotFoundError:
        print(f"Error: Profile file '{profile_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading profile file '{profile_path}': {e}")
        return

    print(f"--- Profiling Analysis for: {profile_path} ---")

    # --- General Cleaning and Sorting Options ---
    # Strip leading path information from filenames for readability
    stats.strip_dirs()

    # Sort by cumulative time spent in the function and its subfunctions
    print(f"\n--- Top {num_stats_to_show} Stats (Sorted by Cumulative Time) ---")
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(num_stats_to_show)

    # Sort by total time spent in the function itself (excluding subfunctions)
    print(f"\n--- Top {num_stats_to_show} Stats (Sorted by Total Time in Function) ---")
    stats.sort_stats(SortKey.TIME) # or SortKey.TOTALTIME
    stats.print_stats(num_stats_to_show)

    # Sort by number of calls
    print(f"\n--- Top {num_stats_to_show} Stats (Sorted by Number of Calls) ---")
    stats.sort_stats(SortKey.CALLS)
    stats.print_stats(num_stats_to_show)

    # --- More Detailed Information (Optional) ---

    # Print callers of a specific function (example)
    # Replace 'your_function_name_here' with a function you're interested in.
    # You might need to find the exact string representation from the output above.
    # function_to_inspect = 'process_and_label_di' # Example
    # print(f"\n--- Callers of '{function_to_inspect}' ---")
    # try:
    #     stats.print_callers(function_to_inspect, num_stats_to_show)
    # except Exception as e:
    #     print(f"Could not get callers for '{function_to_inspect}': {e} (Function name might be slightly different in profiler output)")

    # Print callees of a specific function (example)
    # print(f"\n--- Callees of '{function_to_inspect}' ---")
    # try:
    #     stats.print_callees(function_to_inspect, num_stats_to_show)
    # except Exception as e:
    #     print(f"Could not get callees for '{function_to_inspect}': {e}")

    print("\n--- End of Analysis ---")
    print("Understanding the columns:")
    print("  ncalls: number of calls to the function.")
    print("  tottime: total time spent in the function itself (excluding calls to sub-functions).")
    print("  percall (tottime): tottime / ncalls.")
    print("  cumtime: cumulative time spent in this function and all sub-functions (from invocation till exit).")
    print("  percall (cumtime): cumtime / ncalls.")
    print("  filename:lineno(function): provides the source file, line number, and function name.")

if __name__ == "__main__":
    # Assuming 'mdetector_profile.prof' is in the same directory as this script
    # or provide the full path.
    profile_file = "mdetector_profile.prof"
    
    # You can adjust how many lines of stats you want to see
    num_top_stats = 50 
    
    analyze_profile(profile_file, num_stats_to_show=num_top_stats)

    # --- Example of how to look for specific functions ---
    # stats_obj = pstats.Stats(profile_file)
    # stats_obj.strip_dirs()
    # stats_obj.sort_stats(SortKey.CUMULATIVE)
    # print("\n--- Stats related to 'DepthImage' methods ---")
    # stats_obj.print_stats('depth_image.py') # Print all stats from depth_image.py
    # print("\n--- Stats related to 'MDetector.process_and_label_di' ---")
    # stats_obj.print_stats('process_and_label_di') # Print stats for functions containing this string