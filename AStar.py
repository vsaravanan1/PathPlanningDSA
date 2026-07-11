# Pseudocode from https://youtu.be/-L-WgKMFuhE?si=pInjE2HDrvsEJ23b
# some minor changes/additions were made

import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# priority queue to keep the open section logn
# numpy used for grid

# node class to hold g, h, f, and parent values
class Node:
    def __init__(self, position, parent=None):

        # sets starting vals all to either 0 or arg given
        self.position = position;
        self.parent = parent;
        self.g_cost = 0;
        self.h_cost = 0;
        self.f_cost = 0;

    def __lt__(self, otherNode):

        # overloads < to be for f_cost
        lessThan = (self.f_cost < otherNode.f_cost)
        return lessThan

    def positionEqual(self, otherNode):

        # overloads == to be for position
        positionSame = (self.position == otherNode.position)
        return positionSame





def a_star(grid_num, grid, start, end):
    # convert start and end to nodes
    start = Node(start)
    end = Node(end)

    # force converts grid to numpy for more common input throughout project
    grid = np.array(grid)

    # OPEN (set of nodes to be evaluated)
    # add the start node to OPEN
    openHeap = [start]

    # CLOSED (set of nodes already evaluated)
    closedSet = set()

    # map to keep searching for an open node O(1)
    openMap = {start.position: start}

    #defintion for neighbours
    neighbourDirections = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)];

    # loop
    while openHeap:
        #   current = node in OPEN with the lowest f_cost
        #   remove current from OPEN
        current = heapq.heappop(openHeap)

        #   add current to CLOSED
        closedSet.add(current.position)

        # remove current since its been popped from open
        del openMap[current.position]

        # if current is the target node (path has been found)
        if current.positionEqual(end):
            # prints path if at end
            path = []
            while current:
                path.append(current.position)
                current = current.parent

            if len(path) > 0:
                plotted_path = np.array(path)
                plt.figure(figsize=(10, 10), dpi=300)
                plt.imshow(grid, cmap='gray_r', interpolation='nearest', origin='lower')
                plt.plot(plotted_path[:, 1], plotted_path[:, 0], linewidth=0.25, c='red')
                plt.savefig(f"astar_success_{grid_num}.png", bbox_inches="tight")

            # returns entire path in reverse
            return path[::-1]

        # for each neighbour of the current node
        for rowIndex, colIndex in neighbourDirections:
            # sets row and column of grid to the current heap's neighbours' row and column
            # sets row and column to be the row and col of the neighbour
            row = current.position[0] + rowIndex
            col = current.position[1] + colIndex

            # if neighbour is not traversible or neighbour is in CLOSED
            # also checks if barrier exists there
            if (row < 0 or row >= grid.shape[0]) or (col < 0 or col >= grid.shape[1]) or (grid[row][col] == 1) or ((row, col) in closedSet):

                # skip to the next neighbour
                continue

            # making current neighbour a node
            neighbour = Node((row, col), current)

            # determining cost of movement of g
            # if straightaway (so one direction is not moved, add 10
            if rowIndex == 0 or colIndex == 0:
                moveCost = 10

            # if diagonal, add 14
            else:
                moveCost = 14

            # if new path to neighbour is shorter
            if ((row, col) not in openMap):

                # distance from end point for x and y
                xDistance = abs(row - end.position[0])
                yDistance = abs(col - end.position[1])

                # if distance is shorter horizontally, go diagonally for verticle section and horizontally for remaining
                if xDistance > yDistance:
                    neighbour.h_cost = 14 * yDistance + 10 * (xDistance - yDistance)

                # if distance is shorter vertially, go diagonally for horizontal section and vreticle for remaining
                else:
                    neighbour.h_cost = 14 * xDistance + 10 * (yDistance - xDistance)

                # set f_cost of neighbour
                neighbour.g_cost = current.g_cost + moveCost;
                neighbour.f_cost = neighbour.g_cost + neighbour.h_cost

                # set parent of neighbour to current
                neighbour.parent = current;

                # add neighbour to OPEN
                heapq.heappush(openHeap, neighbour)
                openMap[neighbour.position] = neighbour

            # seperated since neighbour would be treated as a new node with 0 f,g,h cost otherwise
            else:
                # obtaining neighbour
                neighbour = openMap[(row, col)]

                if (current.g_cost + moveCost < neighbour.g_cost):

                    # setting costs and parent same as previous
                    neighbour.g_cost = current.g_cost + moveCost
                    neighbour.f_cost = neighbour.g_cost + neighbour.h_cost
                    neighbour.parent = current

                    # heapify to account for changed g, f, h costs
                    heapq.heapify(openHeap)


#def main():

#    grid = np.load("LDEM_875S_5M.npy")

#    start = (0, 0)
#    end = (99, 99)

#    path = a_star(grid, start, end)

#    if path:
#        path = np.array(path)
#        plt.figure(figsize=(10, 10), dpi=300)
#        plt.imshow(grid, cmap='gray_r', interpolation='nearest', origin='lower')
#        plt.plot(path[:, 1], path[:, 0], linewidth=0.5, c='red')
#        obstacle_count = 0
#        for waypoint in path:
#            r, c = waypoint
#            if grid[r][c] == 1:
#                obstacle_count += 1
#        print(f"Obstacle count in path: {obstacle_count}")
#        #plt.savefig("astar_success.png", bbox_inches="tight")
#       plt.show()
#    else:
#        print("No path found")

#if __name__ == "__main__":
#    main()
