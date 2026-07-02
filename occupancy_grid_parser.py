import os
import matplotlib.pyplot as plt
import numpy as np

class OccupancyGridUtilities:
    def __init__(self):
        self.ds_path = os.path.join(os.path.dirname(__file__), 'Dataset')
        self.dir_list = os.listdir(self.ds_path)
        self.occupancy_grids = self.load_files()
        
    def load_files(self):
        occupancy_grids = []
        for i, np_file in enumerate(self.dir_list):
            file_path = os.path.join(self.ds_path, np_file)
            sub_ds = np.load(file_path)
            print(sub_ds.shape)
            name = f"occupancy_check_{i}.png"
            plt.tight_layout()
            plt.imshow(sub_ds, cmap='gray_r')
            plt.title("0=white, 1=black")
            plt.colorbar()
            plt.savefig(name)
            plt.close()
            occupancy_grids.append(sub_ds)
    
        return occupancy_grids
    
    def __len__(self):
        return len(self.occupancy_grids)

    def shape(self, i):
        return self.occupancy_grids[i].shape
    
    def get_local_path(self, start : np.ndarray, end : np.ndarray, grid_num : int):
        if not self.in_bounds(start, grid_num) or not self.in_bounds(end, grid_num): return np.empty(0, dtype=int)
        if (start == end).all():
            return np.array([start], dtype=int)
        disp = end - start
        step = np.sign(disp)
        
        curr = start.copy()
        trajectory = []
            
        dx, dy = step
        
        if dx == 0 or dy == 0:
            while ((curr != end).any()):
                trajectory.append(curr.copy())
                curr += step
            trajectory.append(end)
            return np.array(trajectory, dtype=int)
        
        def _planning_condition(curr, dx, dy, end):
            curr_copy = np.copy(curr)
            end_copy = np.copy(end)
            
            if dx < 0:
                curr_copy[0] = -1 * curr_copy[0]
                end_copy[0] = -1 * end_copy[0]
            
            if dy < 0:
                curr_copy[1] = -1 * curr_copy[1]
                end_copy[1] = -1 * end_copy[1]

            return (curr_copy <= end_copy).all()
                

        while _planning_condition(curr, dx, dy, end):
            trajectory.append(curr.copy())
            curr += step
        
        if (trajectory[-1] == end).all(): return np.array(trajectory, dtype=int)
        
        last = trajectory[-1].copy()
        
        if last[0] == end[0]:
            while (True):
                last[1] += dy
                trajectory.append(last.copy())
                if last[1] == end[1]: break
        else:
            while (True):
                last[0] += dx
                trajectory.append(last.copy())
                if last[0] == end[0]: break

        return np.array(trajectory, dtype=int)


    def in_bounds(self, point : np.ndarray, grid_num : int):
        x, y = point
        h, v = self.occupancy_grids[grid_num].shape
        return x >= 0 and x < h and y >= 0 and y < v
    

    """ n by 2 array where each row is waypoint (x, y)"""
    def collision_exists(self, trajectory : np.ndarray, grid_num: int):
        if len(trajectory) == 1 or len(trajectory) == 0: return True
        for i in range(trajectory.shape[0]):
            x, y = trajectory[i]
            og = self.occupancy_grids[grid_num]
            if og[x, y] == 1:
                return True            
        return False
    
    def convert_coords_cd(self, coords, grid_num):
        return np.astype(coords * np.array(list(self.occupancy_grids[grid_num].shape)), int)

    def convert_coords_dc(self, coords, grid_num):
        return np.astype(coords / np.array(list(self.occupancy_grids[grid_num].shape)), np.float64)
    
    

    
    

        

            

    
        

        