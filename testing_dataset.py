import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from scipy.spatial import KDTree
import struct
from occupancy_grid_parser import OccupancyGridUtilities
import pdb


# # implement RRT for each of these occupancy grids and test the length of the path generated and time taken

class DynamicKDTree:
    def __init__(self):
        self.tree = None
        self.pending_buffer = []
        self.count = 0
    
    def insert(self, coordinates : list[np.ndarray]):
        self.pending_buffer += coordinates
        if self.count == 0:
            self.tree = KDTree(coordinates)
            self.pending_buffer = []
        elif len(self.pending_buffer) >= 50:
            current_tree_data = self.tree.data.tolist()
            current_tree_data += self.pending_buffer
            self.tree = KDTree(current_tree_data)
            self.pending_buffer = []
        self.count += len(coordinates)
    
    def query(self, coordinate : np.ndarray):
        if self.tree is None:
            raise ValueError("Cannot query an empty DynamicKDTree — call insert() first.")
        d, idx = self.tree.query(coordinate)
        tree_candidate = self.tree.data[idx]
        min_dist = d
        

        for i, pending_candidate in enumerate(self.pending_buffer):
            dist = np.linalg.norm(self.pending_buffer[i] - coordinate, ord=2)
            if dist < min_dist:
                min_dist = dist
                tree_candidate = pending_candidate
                idx = i + len(self.tree.data)
        
        return idx, tree_candidate
    

class RRTPlanner:
    def __init__(self, max_tree_size, grid_num):
        self.search_tree = DynamicKDTree()
        self.obstacles = None
        self.max_tree_size = max_tree_size
        self.og_util = OccupancyGridUtilities()
        self.grid_num = grid_num
        self.start = self.get_start_node(1)
        self.goal = self.get_end_node(1)
        self.search_tree.insert([self.start])
        self.parents = {0: -1,}
        self.nodes = [self.start]

        

    # mode = 0 -> start is closest to (0, 0)
    def get_start_node(self, mode):
        if mode == 0:
            return np.array([0, 0])
        elif mode == 1:
            return np.array([391, 22])
        return np.array([0, 0])

    def get_end_node(self, mode):
        og_shape = np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))
        og_final = og_shape - 1
        if mode == 0:
            return np.array([100, 22])
        return og_final
    
    
    def expand_tree(self, step_size):
        sample_goal = np.random.random_sample() < 0.1
        og_shape = np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))
       
        if not sample_goal:
            new_candidate = np.random.rand(2)  
        else:
            new_candidate = np.astype(self.goal/og_shape, np.float64)
    
        new_candidate_discrete = np.astype(new_candidate * og_shape, int)

        idx, nearest_node = self.search_tree.query(new_candidate_discrete)
        nearest_node = nearest_node.astype(int)
        direction = new_candidate_discrete - nearest_node
    
        # normalized direction
        if np.linalg.norm(direction) == 0:
            return []
        direction_normalized = direction/np.linalg.norm(direction)
        if np.linalg.norm(direction) >= step_size:
            new_candidate = self.og_util.convert_coords_dc(nearest_node, self.grid_num) + step_size * direction_normalized
            new_candidate_discrete = self.og_util.convert_coords_cd(new_candidate, self.grid_num)
        # evaluate collision between marker and obstacle
        if ((new_candidate_discrete == nearest_node).all()):
            return []

        local_path = self.og_util.get_local_path(nearest_node, new_candidate_discrete, self.grid_num)
        if len(local_path) == 0: return []
        if not self.og_util.collision_exists(local_path, self.grid_num):
             for i in range(1, len(local_path)):
                self.nodes.append(local_path[i])
                self.search_tree.insert([local_path[i]])
                self.parents[self.search_tree.count - 1] = idx
                idx = self.search_tree.count - 1

                if np.linalg.norm(local_path[i] - self.goal) == 0:
                    path = []
                    candidate_node = self.search_tree.count - 1
                    while candidate_node != -1:
                        path.append(self.nodes[candidate_node])
                        candidate_node = self.parents[candidate_node]

                    path.reverse()  
                    
                    return path
                
        return []
    
    def run_rrt(self):
        i = 0
        step_size = 0.005
        og = self.og_util.occupancy_grids[self.grid_num]
        while self.search_tree.count < self.max_tree_size and i < 10 * self.max_tree_size:
            i += 1
            path = np.array(self.expand_tree(step_size))
            if len(path) > 0:
                print(path)
                plt.figure(figsize=(10, 10), dpi=300)
                plt.imshow(og, cmap='gray_r', interpolation='nearest', origin='lower')
                plt.plot(path[:,1], path[:,0], linewidth=0.5, c='red')
                obstacle_count = 0
                for waypoint in path:
                    r, c = waypoint
                    if og[r, c] == 1:
                        obstacle_count += 1
                print(f"Obstacle count in path: {obstacle_count}")
                plt.scatter(self.goal[1], self.goal[0], s=50, c='blue', marker='*')
                plt.savefig('rrt_success.png')
                plt.close()
                return 
        print("failed")
        plt.imshow(og, cmap='gray_r', interpolation='none', origin='lower')
        nodes_arr = np.array(self.nodes)
        plt.scatter(nodes_arr[:, 1], nodes_arr[:, 0], s=1, c='red')
        plt.scatter(self.goal[1], self.goal[0], s=50, c='blue', marker='*')
        plt.savefig("rrt_debug.png")
        plt.close()

        

def main():
    planner = RRTPlanner(40000, 0)
    planner.run_rrt()

if __name__ == "__main__":
    main()
