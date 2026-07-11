from comparison import compare_on_map
from occupancy_grid_parser import OccupancyGridUtilities

def run_all_experiments(og_util, num_grids=12):
    results = []
    for grid_num in range(num_grids):
        print(f"Running comparison on map {grid_num}...")
        result = compare_on_map(og_util, grid_num)
        results.append(result)
    return results

def summarize(results, algos=("astar", "rrt", "rrtstar")):
    summary = {}
    total = len(results)
    for algo in algos:
        successes = [r[algo] for r in results if r[algo]["success"]]
        success_count = len(successes)
        avg_efficiency = (
            sum(s["efficiency"] for s in successes) / success_count
            if success_count > 0
            else None
        )
        avg_time = (
            sum(s["time_sec"] for s in successes) / success_count
            if success_count > 0
            else None
        )

        summary[algo] = {
            "success_rate": success_count / total,
            "num_successes": success_count,
            "total": total,
            "avg_efficiency": avg_efficiency,
            "avg_time": avg_time,
        }
    return summary

def print_summary(summary):
    print("\n" + "=" * 60)
    print("SUMMARY ACROSS ALL MAPS")
    print("=" * 60)
    for algo, stats in summary.items():
        print(f"\n{algo.upper()}")
        print(f"  Success rate : {stats['num_successes']}/{stats['total']} "
              f"({stats['success_rate']*100:.1f}%)")
        if stats["avg_efficiency"] is not None:
            print(f"  Avg efficiency : {stats['avg_efficiency']:.2f}")
            print(f"  Avg time (s)   : {stats['avg_time']:.3f}")
        else:
            print("  No successful runs to average.")


