import time 
import os
import sys
import numpy as np
from AStar import a_star
from rrt_planner import RRTStarPlanner, RRTPlanner
from occupancy_grid_parser import OccupancyGridUtilities
from get_start_goal import get_start_goal

def astar_test(grid, start, goal): #astar timing
    ts = time.perf_counter()
    path = a_star(grid, tuple(start), tuple(goal))
    elapsed = time.perf_counter() - ts
    return path, elapsed

def rrtstar_test(og_util, grid_num, start, goal, max_tree_size=3000, max_attempts=4): #rstar timing
    ts = time.perf_counter()
    path = None
    for j in range(max_attempts):
        plan = RRTStarPlanner(max_tree_size, grid_num, og_util, start, goal, random_state=10 + j)
        path = plan.run(step_size = 0.05, radius = 10 - j)
        if path is not None and len(path) > 0:
            break
    elapsed = time.perf_counter() - ts
    return path, elapsed



def rrt_test(og_util, grid_num, start, goal, max_tree_size=3000, max_attempts=4): #rstar timing 
    ts = time.perf_counter()
    path = None
    for j in range(max_attempts):
        plan = RRTPlanner(max_tree_size, grid_num, og_util, start, goal, random_state=10 + j)
        path = plan.run(step_size = 0.05)
        if path is not None and len(path) > 0:
            break
    elapsed = time.perf_counter() - ts
    return path, elapsed

def path_length(path):
    if path is None or len(path) < 2:
        return None
    arr = np.array(path)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))

def straight_line_distance(start, goal):
    return float(np.linalg.norm(np.array(goal) - np.array(start)))

def efficiency(path, start, goal):
    length = path_length(path)
    if length is None or length == 0:
        return None
    return straight_line_distance(start, goal) / length

def compare_on_map(og_util, grid_num):
    grid = og_util.occupancy_grids[grid_num]
    start, goal = get_start_goal(og_util,grid_num)
    astar_path, astar_time = astar_test(grid, start, goal)
    rrtstar_path, rrtstar_time = rrtstar_test(og_util, grid_num, start, goal)
    rrt_path, rrt_time = rrt_test(og_util, grid_num, start, goal)

    
    return {
        "grid_num": grid_num,
        "start": start,
        "goal": goal,
        "astar": {
            "time_sec": astar_time,
            "success": astar_path is not None,
            "path_length": path_length(astar_path),
            "efficiency": efficiency(astar_path, start, goal) if astar_path else None,
        },
        "rrt": {
            "time_sec": rrt_time, 
            "success": rrt_path is not None,
            "path_length": path_length(rrt_path),
            "efficiency": efficiency(rrt_path, start, goal) if rrt_path is not None else None,
        },
        "rrtstar": {
            "time_sec": rrtstar_time, 
            "success": rrtstar_path is not None,
            "path_length": path_length(rrtstar_path),
            "efficiency": efficiency(rrtstar_path, start, goal) if rrtstar_path is not None else None,
        },
    }

def format(name, stats):
    if not stats["success"]:
        return f"{name} failed to find a path."
    return (f"  {name:<8} length = {stats['path_length']:>8.1f}   "
            f"efficiency = {stats['efficiency']:.2f}   time = {stats['time_sec']:.3f}s") 

def print_comparison(result):
    start = tuple(int(x) for x in result['start'])
    goal = tuple(int(x) for x in result['goal'])
    print(f"\nMap {result['grid_num']} | start={start}  goal={goal}:")
    print("-" * 60)
    print(format("A*", result["astar"]))
    print(format("RRT*", result["rrtstar"]))
    print(format("RRT", result["rrt"]))


if __name__ == "__main__":
    og_util = OccupancyGridUtilities()
    r1 = compare_on_map(og_util, 1)
    r2 = compare_on_map(og_util, 2)
    r3 = compare_on_map(og_util, 3)
    r4 = compare_on_map(og_util, 4)
    r5 = compare_on_map(og_util, 5)
    print_comparison(r1)
    print_comparison(r2)
    print_comparison(r3)
    print_comparison(r4)
    print_comparison(r5)