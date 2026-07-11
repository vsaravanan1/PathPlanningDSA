import time 
import os
import sys
import numpy as np
from AStar import a_star
from rrt_planner import RRTStarPlanner, RRTPlanner
from occupancy_grid_parser import OccupancyGridUtilities
from get_start_goal import get_start_goal

def astar_test(grid_num, grid, start, goal): 
    ts = time.perf_counter()
    path = a_star(grid_num, grid, tuple(start), tuple(goal))
    elapsed = time.perf_counter() - ts
    return path, elapsed

def rrtstar_test(og_util, grid_num, start, goal, max_tree_size=20000, max_attempts=4): 
    ts = time.perf_counter()
    path = None
    for j in range(max_attempts):
        plan = RRTStarPlanner(max_tree_size, grid_num, og_util, start, goal, random_state=10 + j)
        path = plan.run(0.05, 10 - j)
        if path is not None and len(path) > 0:
            break
    elapsed = time.perf_counter() - ts
    return path, elapsed

def rrt_test(og_util, grid_num, start, goal, max_tree_size=20000, max_attempts=4):  
    ts = time.perf_counter()
    path = None
    for j in range(max_attempts):
        plan = RRTPlanner(max_tree_size, grid_num, og_util, start, goal, random_state=10 + j)
        path = plan.run(0.05)
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

    astar_path, astar_time = astar_test(grid_num, grid, start, goal)
    rrtstar_path, rrtstar_time = rrtstar_test(og_util, grid_num, start, goal)
    rrt_path, rrt_time = rrt_test(og_util, grid_num, start, goal)

    astar_success = astar_path is not None
    rrt_success = rrt_path is not None
    rrtstar_success = rrtstar_path is not None

    astar_eff = efficiency(astar_path, start, goal)
    rrt_eff = efficiency(rrt_path, start, goal)
    rrtstar_eff = efficiency(rrtstar_path, start, goal)
    

    astar_info_dict = {"time": astar_time, "success": astar_success, "path_length": path_length(astar_path), "efficiency": astar_eff}
    rrt_info_dict = {"time": rrt_time, "success": rrt_success, "path_length": path_length(rrt_path), "efficiency": rrt_eff}
    rrtstar_info_dict = {"time": rrtstar_time, "success": rrtstar_success, "path_length": path_length(rrtstar_path), "efficiency": rrtstar_eff}


    comparison_dict = {}
    comparison_dict["endpoints"] = (start, goal)
    comparison_dict["astar"] = astar_info_dict
    comparison_dict["rrt"] = rrt_info_dict
    comparison_dict["rrtstar"] = rrtstar_info_dict
    
    return comparison_dict


def print_comparison(comparison_dict, grid_num):
    start, goal = comparison_dict["endpoints"]
    print(f"\nMap {grid_num} - Start: {start},  Goal: {goal}")
    algos = ["astar", "rrt", "rrtstar"]

    best_time = np.inf
    best_time_algo = None
    best_efficiency = 0.0
    best_efficiency_algo = None

    for algo in algos:
        algo_info_dict = comparison_dict[algo]
        if algo_info_dict["success"]:
            print(f"{algo}: Time Taken - {algo_info_dict["time"]}, Efficiency (relative to straight line) - {algo_info_dict["efficiency"]}")
            if algo_info_dict["time"] < best_time:
                best_time_algo = algo
                best_time = algo_info_dict["time"]
            if algo_info_dict["efficiency"] > best_efficiency:
                best_efficiency_algo = algo
                best_efficiency = algo_info_dict["efficiency"]
        else:
            print(f"{algo}: Unsuccessful. Could not find path.")
    
    if best_time_algo and best_efficiency_algo:
        print()
        print(f"The algorithm that found a successful path in the least time was: {best_time_algo}.")
        print(f"The algorithm that found the shortest path was: {best_efficiency_algo}.")
        print()
    else:
        print()
        print("None of the algorithms could find a viable path.")
        print()


def get_average_performance(map_result_dicts):
    success_count = [0, 0, 0]
    average_time = [0.0, 0.0, 0.0]
    average_efficiency = [0.0, 0.0, 0.0]

    for comparison_dict in map_result_dicts:
        astar_dict = comparison_dict["astar"]
        rrt_dict = comparison_dict["rrt"]
        rrtstar_dict = comparison_dict["rrtstar"]

        if astar_dict["success"]: 
            success_count[0] += 1
            average_time[0] += astar_dict["time"]
            average_efficiency[0] += astar_dict["efficiency"]
        
        if rrt_dict["success"]:
            success_count[1] += 1
            average_time[1] += rrt_dict["time"]
            average_efficiency[1] += rrt_dict["efficiency"]
        
        if rrtstar_dict["success"]:
            success_count[2] += 1
            average_time[2] += rrtstar_dict["time"]
            average_efficiency[2] += rrtstar_dict["efficiency"]

        for i in range(len(success_count)):
            if success_count[i] == 0:
                average_efficiency[i] = -1.0
                average_time[i] = -1.0
            else:
                average_efficiency[i] = average_efficiency[i]/success_count[i]
                average_time[i] = average_time[i]/success_count[i]

    return success_count, average_time, average_efficiency

def print_summary_stats(map_result_dicts):
    success_count, average_time, average_efficiency = get_average_performance(map_result_dicts)
    algos = ["astar", "rrt", "rrtstar"]

    print()
    print("Path Planning Algorithm Performance Summary")
    for i in range(len(success_count)):
        if success_count[i] > 0:
            print(f"{algos[i]}:", f"success rate: {success_count[i]}/{len(map_result_dicts)}, average time: f{average_time[i]}, average efficiency: {average_efficiency[i]}")
        else:
            print(f"{algos[i]}:", f"success_rate: 0/{len(map_result_dicts)}")



if __name__ == "__main__":
    og_util = OccupancyGridUtilities()
    comparison_dicts = []
    for i in range(len(og_util.occupancy_grids)):
        map_result_dict = compare_on_map(og_util, i)
        print_comparison(map_result_dict, i)
        comparison_dicts.append(map_result_dict)
    print_summary_stats(comparison_dicts)
    
