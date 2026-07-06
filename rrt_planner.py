import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from scipy.spatial import KDTree
import struct
from occupancy_grid_parser import OccupancyGridUtilities


# implement RRT for each of these occupancy grids and test the length of the path generated and time taken

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
    
    def query_ball_point(self, coordinate : np.ndarray, radius):
        if self.tree is None:
            raise ValueError("Cannot query an empty DynamicKDTree — call insert() first.")
        
        idx, tree_candidate = self.query(coordinate)
        dist = np.linalg.norm(tree_candidate - coordinate)
        if radius < dist:
            return [idx], [tree_candidate]
        else:
            candidate_points = []
            indices = self.tree.query_ball_point(coordinate, radius)
            for index in indices:
                candidate_points.append(self.tree.data[index])

            for i, pending_point in enumerate(self.pending_buffer):
                if np.linalg.norm(pending_point - coordinate) <= radius:
                    candidate_points.append(pending_point)
                    indices.append(len(self.tree.data) + i)
            
            return indices, candidate_points
    

    

class RRTPlanner:
    def __init__(self, max_tree_size, grid_num, og_util, start : np.array, goal : np.array):
        self.search_tree = DynamicKDTree()
        self.max_tree_size = max_tree_size
        self.og_util = og_util
        self.grid_num = grid_num
        self.start = start
        self.goal = goal
        self.search_tree.insert([self.start])
        self.parents = {0: -1,}
        self.nodes = [self.start]
    
    def expand_tree(self, step_size):
        sample_goal = np.random.random_sample() < 0.2
        og_shape = np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))
       
        if not sample_goal:
            new_candidate = np.random.rand(2)  
        else:
            new_candidate = (self.goal/og_shape).astype(np.float64)
    
        new_candidate_discrete = (new_candidate * og_shape).astype(int)

        idx, nearest_node = self.search_tree.query(new_candidate_discrete)
        nearest_node = nearest_node.astype(int)
        direction = new_candidate_discrete - nearest_node


        # normalized direction
        if np.linalg.norm(direction) == 0:
            return []
        direction_normalized = direction/np.linalg.norm(direction)
        if np.linalg.norm(direction/np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))) >= step_size:
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

                if np.linalg.norm(local_path[i] - self.goal) == 0.0:
                    path = []
                    candidate_node = self.search_tree.count - 1
                    while candidate_node != -1:
                        path.append(self.nodes[candidate_node])
                        candidate_node = self.parents[candidate_node]

                    path.reverse()  

                    return path
        else:
            # consider only the segment of the path leading up to the first collision
            for i in range(1, len(local_path)):
                if self.og_util.occupancy_grids[self.grid_num][tuple(local_path[i].tolist())] == 1:
                    break
                self.nodes.append(local_path[i])
                self.search_tree.insert([local_path[i]])
                self.parents[self.search_tree.count - 1] = idx
                idx = self.search_tree.count - 1

                if np.linalg.norm(local_path[i] - self.goal) == 0.0:
                    path = []
                    candidate_node = self.search_tree.count - 1
                    while candidate_node != -1:
                        path.append(self.nodes[candidate_node])
                        candidate_node = self.parents[candidate_node]

                    path.reverse()  
                    
                    return path

                
        return []
    
    def run(self):
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
                plt.savefig(f'rrt_star_success_{self.grid_num}.png')
                plt.close()
                return 
        print("failed")
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(og, cmap='gray_r', interpolation='none', origin='lower')
        nodes_arr = np.array(self.nodes)
        plt.scatter(nodes_arr[:, 1], nodes_arr[:, 0], s=1, c='red')
        plt.scatter(self.goal[1], self.goal[0], s=50, c='blue', marker='*')
        plt.savefig(f"rrt_star_debug_{self.grid_num}.png")
        plt.close()

class RRTStarPlanner:
    def __init__(self, max_tree_size, grid_num, og_util : OccupancyGridUtilities, start, goal):
        self.search_tree = DynamicKDTree()
        self.max_tree_size = max_tree_size
        self.og_util = og_util
        self.grid_num = grid_num
        self.start = start
        self.goal = goal
        self.search_tree.insert([self.start])
        self.parents = {0: -1,}
        self.nodes = [self.start]
        self.total_costs = {0: 0,}
        self.goal_idx = -1
    
    def expand_tree(self, step_size, radius):
        sample_goal = np.random.random_sample() < 0.2
        og_shape = np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))
        
        if not sample_goal:
            new_candidate = np.random.rand(2)  
            new_candidate_discrete = (new_candidate * og_shape).astype(int)
        else:
            new_candidate_discrete = self.goal.copy()
        

        idx, nearest_node = self.search_tree.query(new_candidate_discrete)
        nearest_node = nearest_node.astype(int)
        direction = new_candidate_discrete - nearest_node

        dist_pixels = np.linalg.norm(direction)
        if dist_pixels == 0:
            return []
            
        max_step_pixels = max(1, int(step_size * max(og_shape)))
        
        if dist_pixels > max_step_pixels:
            direction_normalized = direction / dist_pixels
            new_candidate_discrete = nearest_node + direction_normalized * max_step_pixels
            new_candidate_discrete = new_candidate_discrete.astype(int)
            
        if ((new_candidate_discrete == nearest_node).all()):
            return []


        indices, nearby_nodes = self.search_tree.query_ball_point(new_candidate_discrete, radius)
        nearby_nodes = [node.astype(int) for node in nearby_nodes]
        
        chosen_candidate_cost = np.inf
        chosen_path = None
        idx = -1

        for i, node in enumerate(nearby_nodes):    
            local_path = self.og_util.get_local_path(node, new_candidate_discrete, self.grid_num)
            if len(local_path) == 0: continue
            # if collision doesn't exist, calculate cost
            if not self.og_util.collision_exists(local_path, self.grid_num):
                steps = local_path[1:] - local_path[0:-1]
                step_norms = np.linalg.norm(steps, axis=1)
                candidate_cost = self.total_costs[indices[i]] + np.sum(step_norms)
                if candidate_cost < chosen_candidate_cost:
                    chosen_candidate_cost = candidate_cost
                    chosen_path = local_path
                    idx = indices[i]
        
        if idx == -1: return []
        
        self.nodes.append(chosen_path[-1])
        self.search_tree.insert([chosen_path[-1]])
        self.parents[self.search_tree.count - 1] = idx
        self.total_costs[self.search_tree.count - 1] = chosen_candidate_cost

        # neighbor modification
        for i, node in enumerate(nearby_nodes):
            node_idx = indices[i]
            if node_idx == idx: continue
            nearby_to_curr_path = self.og_util.get_local_path(chosen_path[-1], self.nodes[node_idx], self.grid_num)
            if not self.og_util.collision_exists(nearby_to_curr_path, self.grid_num):
                steps = nearby_to_curr_path[1:] - nearby_to_curr_path[0:-1]
                step_norms = np.linalg.norm(steps, axis=1)
                path_cost = np.sum(step_norms)
                new_node_idx = self.search_tree.count - 1
                candidate_total_cost = path_cost + self.total_costs[new_node_idx]
                if candidate_total_cost < self.total_costs[node_idx]:
                    self.total_costs[node_idx] = candidate_total_cost
                    self.parents[node_idx] = new_node_idx
        
        path = []
        for i in range(chosen_path.shape[0]):
            if (chosen_path[i] == self.goal).all():
                local_path = self.og_util.get_local_path(chosen_path[0], chosen_path[i], self.grid_num)
                local_path_ex = local_path[1:,:]
                for j in range(local_path_ex.shape[0]):
                    path.append(local_path_ex[local_path_ex.shape[0] - 1 - j])
                
                current_node_idx = self.parents[self.search_tree.count-1]
                parent_node_idx = self.parents[current_node_idx]
                while current_node_idx != 0:
                    current_node = self.nodes[current_node_idx]
                    parent_node = self.nodes[parent_node_idx]
                    local_path = self.og_util.get_local_path(parent_node, current_node, self.grid_num)
                    if parent_node_idx == 0:
                        local_path_ex = local_path
                    else:
                        local_path_ex = local_path[:-1,:]
                    for j in range(local_path_ex.shape[0]):
                        path.append(local_path_ex[local_path_ex.shape[0] - 1 - j])
                    
                    current_node_idx = self.parents[current_node_idx]
                    parent_node_idx = self.parents[parent_node_idx]

                path.reverse()
                return path
        
        return []

    def expand_tree(self, step_size, radius):
        sample_goal = np.random.random_sample() < 0.2
        og_shape = np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))
        
        if not sample_goal:
            new_candidate = np.random.rand(2)  
        else:
            new_candidate = (self.goal/og_shape).astype(np.float64)
    
        new_candidate_discrete = (new_candidate * og_shape).astype(int)
        
        # find nearest node and steer towards it
        idx, nearest_node = self.search_tree.query(new_candidate_discrete)
        nearest_node = nearest_node.astype(int)
        direction = new_candidate_discrete - nearest_node

        # normalized direction
        if np.linalg.norm(direction) == 0:
            return []
        direction_normalized = direction/np.linalg.norm(direction)
        if np.linalg.norm(direction/np.array(list(self.og_util.occupancy_grids[self.grid_num].shape))) >= step_size:
            new_candidate = self.og_util.convert_coords_dc(nearest_node, self.grid_num) + step_size * direction_normalized
            new_candidate_discrete = self.og_util.convert_coords_cd(new_candidate, self.grid_num)
        # evaluate collision between marker and obstacle
        if ((new_candidate_discrete == nearest_node).all()):
            return []


        indices, nearby_nodes = self.search_tree.query_ball_point(new_candidate_discrete, radius)
        nearby_nodes = [node.astype(int) for node in nearby_nodes]
        
        chosen_candidate_cost = np.inf
        chosen_path = None
        idx = -1

        for i, node in enumerate(nearby_nodes):    
            local_path = self.og_util.get_local_path(node, new_candidate_discrete, self.grid_num)
            if len(local_path) == 0: continue
            # if collision doesn't exist, calculate cost
            if not self.og_util.collision_exists(local_path, self.grid_num):
                steps = local_path[1:] - local_path[0:-1]
                step_norms = np.linalg.norm(steps, axis=1)
                candidate_cost = self.total_costs[indices[i]] + np.sum(step_norms)
                if candidate_cost < chosen_candidate_cost:
                    chosen_candidate_cost = candidate_cost
                    chosen_path = local_path
                    idx = indices[i]
        
        if idx == -1: return []
        
        self.nodes.append(chosen_path[-1])
        self.search_tree.insert([chosen_path[-1]])
        self.parents[self.search_tree.count - 1] = idx
        self.total_costs[self.search_tree.count - 1] = chosen_candidate_cost

        # neighbor modification
        for i, node in enumerate(nearby_nodes):
            node_idx = indices[i]
            if node_idx == idx: continue
            nearby_to_curr_path = self.og_util.get_local_path(chosen_path[-1], self.nodes[node_idx], self.grid_num)
            if not self.og_util.collision_exists(nearby_to_curr_path, self.grid_num):
                steps = nearby_to_curr_path[1:] - nearby_to_curr_path[0:-1]
                step_norms = np.linalg.norm(steps, axis=1)
                path_cost = np.sum(step_norms)
                new_node_idx = self.search_tree.count - 1
                candidate_total_cost = path_cost + self.total_costs[new_node_idx]
                if candidate_total_cost < self.total_costs[node_idx]:
                    self.total_costs[node_idx] = candidate_total_cost
                    self.parents[node_idx] = new_node_idx
        
        path = []
        for i in range(chosen_path.shape[0]):
            if (chosen_path[i] == self.goal).all():
                local_path = self.og_util.get_local_path(chosen_path[0], chosen_path[i], self.grid_num)
                local_path_ex = local_path[1:,:]
                for j in range(local_path_ex.shape[0]):
                    path.append(local_path_ex[local_path_ex.shape[0] - 1 - j])
                
                current_node_idx = self.parents[self.search_tree.count-1]
                parent_node_idx = self.parents[current_node_idx]
                while current_node_idx != 0:
                    current_node = self.nodes[current_node_idx]
                    parent_node = self.nodes[parent_node_idx]
                    local_path = self.og_util.get_local_path(parent_node, current_node, self.grid_num)
                    if parent_node_idx == 0:
                        local_path_ex = local_path
                    else:
                        local_path_ex = local_path[:-1,:]
                    for j in range(local_path_ex.shape[0]):
                        path.append(local_path_ex[local_path_ex.shape[0] - 1 - j])
                    
                    current_node_idx = self.parents[current_node_idx]
                    parent_node_idx = self.parents[parent_node_idx]

                path.reverse()
                return path
        
        return []

    def run(self):
        i = 0
        step_size = 0.005
        og = self.og_util.occupancy_grids[self.grid_num]
        saved_path = None
        while self.search_tree.count < self.max_tree_size and i < 10 * self.max_tree_size:
            i += 1
            path = np.array(self.expand_tree(step_size, 10))
            if len(path) > 0:
                saved_path = path
        
        if saved_path is not None:
            plt.figure(figsize=(10, 10), dpi=300)
            plt.imshow(og, cmap='gray_r', interpolation='nearest', origin='lower')
            plt.plot(saved_path[:,1], saved_path[:,0], linewidth=0.5, c='red')
            obstacle_count = 0
            for waypoint in saved_path:
                r, c = waypoint
                if og[r, c] == 1:
                    obstacle_count += 1
            print(f"Obstacle count in path: {obstacle_count}")
            plt.scatter(self.goal[1], self.goal[0], s=50, c='blue', marker='*')
            plt.savefig(f'rrt_star_success_{self.grid_num}.png')
            plt.close()
            return
        else:
            print("failed")
            plt.figure(figsize=(10, 10), dpi=300)
            plt.imshow(og, cmap='gray_r', interpolation='none', origin='lower')
            nodes_arr = np.array(self.nodes)
            plt.scatter(nodes_arr[:, 1], nodes_arr[:, 0], s=1, c='red')
            plt.scatter(self.goal[1], self.goal[0], s=50, c='blue', marker='*')
            plt.savefig(f"rrt_star_debug_{self.grid_num}.png")
            plt.close()

        


# def main():
#     planner = RRTPlanner(60000, 0)
#     planner.run_rrt()

# if __name__ == "__main__":
#     main()
