import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.dirname("__file__"))
sys.path.append(parent_dir)

from bfs_floodfill import bfs_floodfill, double_bfs
from rrt_planner import RRTPlanner, RRTStarPlanner, DynamicKDTree
from occupancy_grid_parser import OccupancyGridUtilities


def get_start_goal(og_util: OccupancyGridUtilities, grid_num, manual=True):
    og = og_util.occupancy_grids[grid_num]
    
    if manual:
        start = np.array([0, 0])
        end = np.array(og.shape) - 1

        return start, end
    else:
        _, pockets = bfs_floodfill(og_util, grid_num)

        pockets.sort(key = lambda a : np.sum(a), reverse=True)
        filtered_pockets = pockets[:5]
        
        max_norm = 0
        for mask in filtered_pockets:
            start_point_c, end_point_c = double_bfs(og_util, grid_num, mask)
            if np.linalg.norm(np.array(end_point_c) - np.array(start_point_c)) > max_norm:
                start, end = start_point_c, end_point_c
                max_norm = np.linalg.norm(np.array(end) - np.array(start))

        return start, end
        


def main():
    og_util = OccupancyGridUtilities()
    for i in range(len(og_util.occupancy_grids)):
        start, goal = get_start_goal(og_util, i, False)
        print(start, goal, og_util.occupancy_grids[i][tuple(start)], og_util.occupancy_grids[i][tuple(goal)])
        j = 0
        random_state = 10
        while j < 4:
            planner = RRTStarPlanner(20000, i, og_util, start, goal, random_state+j)
            result = planner.run(0.005*(j+1), 10-j)
            if (result is not None and len(result) > 0): break
            j += 1

if __name__ == "__main__":
    main()