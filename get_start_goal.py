import numpy as np
from bfs_floodfill import bfs_floodfill, double_bfs
from occupancy_grid_parser import OccupancyGridUtilities


def get_start_goal(og_util, grid_num, num_regions=5):
        _, pockets = bfs_floodfill(og_util, grid_num)
        pockets.sort(key=lambda a: np.sum(a), reverse=True)
        large_pockets = pockets[:num_regions]

        best_start, best_goal, best_dist = None, None, -1
        for mask in large_pockets:
            start, goal = double_bfs(og_util, grid_num, mask)
            dist = np.linalg.norm(goal - start)
            if dist > best_dist:
                best_start, best_goal, best_dist = start, goal, dist

        return best_start, best_goal
