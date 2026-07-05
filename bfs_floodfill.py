import queue
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
from occupancy_grid_parser import OccupancyGridUtilities


"""
Breadth-First Search Flood Fill algorithm for start and end node identification from occupancy grid.
"""
def bfs_floodfill(og_util : OccupancyGridUtilities, grid_num : int):
    og = og_util.occupancy_grids[grid_num]
    fs_identifiers = np.ones_like(og, dtype=int) * -1
    to_visit = queue.Queue()
    fs_idx = 0
    for i in range(og.shape[0]):
        for j in range(og.shape[1]):
            if og[i, j] == 1 or fs_identifiers[i, j] != -1:
                continue
            to_visit.put((i, j))
            fs_identifiers[i, j] = fs_idx
            while not to_visit.empty():
                i, j = to_visit.get()
                neighbors = [(i-1, j-1), (i-1, j+1), (i, j-1), (i+1, j-1), (i-1, j), (i+1, j), (i, j+1), (i+1, j+1)]
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < og.shape[0] and 0 <= neighbor[1] < og.shape[1] and og[neighbor] == 0 and fs_identifiers[neighbor] == -1:
                        to_visit.put(neighbor)
                        fs_identifiers[neighbor] = fs_idx
            fs_idx += 1
    
    pockets = []
    for element in np.unique(fs_identifiers):
        pockets.append(np.where(fs_identifiers == element))
    return fs_identifiers, pockets

def double_bfs(og_util : OccupancyGridUtilities, grid_num : int, mask : np.array[bool]):
    og = og_util.occupancy_grids[grid_num]
    valid_points = np.argwhere(mask)
    random.seed(98)
    sampled_row_idx  = random.randint(0, og.shape[0]-1)
    random_point = tuple(valid_points[sampled_row_idx].tolist())
    start_point = None
    end_point = None
    visited = np.ones_like(mask, dtype=int) * -1
    
    # first bfs from random point to start point
    to_visit = queue.Queue()
    to_visit.put(random_point)
    visited[random_point] = 1
    while not to_visit.empty():
        i, j = to_visit.get()
        neighbors = [(i+1, j), (i-1, j), (i+1, j-1), (i, j-1), (i-1, j-1), (i+1, j+1), (i, j+1), (i-1, j+1)]
        for neighbor in neighbors:
            # if neighbor is within bounds 
            if 0 <= neighbor[0] < og.shape[0] and 0 <= neighbor[1] < og.shape[1] and mask[neighbor] and visited[neighbor] == -1:
                visited[neighbor] = 1
                to_visit.put(neighbor)
                
    start_point = (i, j)

    # second bfs from start point to end point
    visited = np.ones_like(mask, dtype=int) * -1
    to_visit.put(start_point)
    visited[start_point] = 1
    while not to_visit.empty():
        (i, j) = to_visit.get()
        neighbors = [(i+1, j), (i-1, j), (i+1, j-1), (i, j-1), (i-1, j-1), (i+1, j+1), (i, j+1), (i-1, j+1)]
        for neighbor in neighbors:
            # if neighbor is within bounds 
            if 0 <= neighbor[0] < og.shape[0] and 0 <= neighbor[1] < og.shape[1] and mask[neighbor] and visited[neighbor] == -1:
                visited[neighbor] = 1
                to_visit.put(neighbor)
    
    end_point = (i, j)
    
    return start_point, end_point


def main():
    # https://stackoverflow.com/questions/19586828/drawing-grid-pattern-in-matplotlib
    og_util = OccupancyGridUtilities()
    og = og_util.occupancy_grids[0]
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    fs_id, _ = bfs_floodfill(og_util, 0)
    max_len = 0
    mask = None
    for id in np.unique(fs_id):
        if len(fs_id == id) > max_len:
            length = len(fs_id == id)
            if length > max_len:
                mask = (fs_id == id)
                max_len = length

    print(max_len)
    valid_indices = np.argwhere(mask)
    print(valid_indices)

    start_point, end_point = double_bfs(og_util, 0, mask)
    print(f"Start: {start_point}")
    print(f"End: {end_point}")


if __name__ == "__main__":
    main()


        
                    