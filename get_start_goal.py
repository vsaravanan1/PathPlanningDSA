import numpy as np
from bfs_floodfill import bfs_floodfill, double_bfs
from occupancy_grid_parser import OccupancyGridUtilities


def get_start_goal(og_util, grid_num, num_pockets_to_check=5):
        _, pockets = bfs_floodfill(og_util, grid_num)
        pockets.sort(key=lambda mask: np.sum(mask), reverse=True)
        top_pockets = pockets[:num_pockets_to_check]

        best_start, best_goal, best_dist = None, None, -1
        for mask in top_pockets:
            start, goal = double_bfs(og_util, grid_num, mask)
            dist = np.linalg.norm(goal - start)
            if dist > best_dist:
                best_start, best_goal, best_dist = start, goal, dist

        return best_start, best_goal

if __name__ == "__main__":
    og_util = OccupancyGridUtilities()
    for grid_num in range(len(og_util)):
        start, goal = get_start_goal(og_util, grid_num)
        dist = np.linalg.norm(goal - start)
        print(f"map {grid_num}: start={start}, goal={goal}, distance={dist:.1f}")