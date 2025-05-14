# src/utils/profiling.py
import time
import functools
from collections import defaultdict

# Global stats dictionary to collect timing data
timing_stats = defaultdict(list)

def timeit(func):
    """
    Decorator to time function execution and collect statistics.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Store the timing
        timing_stats[func.__name__].append(elapsed)
        
        return result
    return wrapper

def print_timing_stats(sort_by='mean', top_n=10):
    """
    Print summary of timing statistics.
    
    Args:
        sort_by: 'mean', 'total', 'calls', or 'max'
        top_n: Number of functions to show
    """
    if not timing_stats:
        print("No timing data collected.")
        return
    
    # Process statistics
    stats = []
    for func_name, times in timing_stats.items():
        total_time = sum(times)
        num_calls = len(times)
        mean_time = total_time / num_calls if num_calls > 0 else 0
        max_time = max(times) if times else 0
        
        stats.append({
            'function': func_name,
            'calls': num_calls,
            'total': total_time,
            'mean': mean_time,
            'max': max_time
        })
    
    # Sort by requested metric
    stats.sort(key=lambda x: x[sort_by], reverse=True)
    
    # Print results
    print(f"\n{'=' * 80}")
    print(f"TIMING STATISTICS (sorted by {sort_by}, showing top {top_n})")
    print(f"{'=' * 80}")
    print(f"{'FUNCTION':<50} {'CALLS':>8} {'TOTAL(s)':>10} {'MEAN(s)':>10} {'MAX(s)':>10}")
    print(f"{'-' * 80}")
    
    for i, stat in enumerate(stats[:top_n]):
        print(f"{stat['function']:<50} {stat['calls']:>8} {stat['total']:>10.4f} {stat['mean']:>10.4f} {stat['max']:>10.4f}")
    
    print(f"{'=' * 80}\n")

def reset_timing_stats():
    """Clear all collected timing statistics."""
    timing_stats.clear()