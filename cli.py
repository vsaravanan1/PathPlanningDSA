from occupancy_grid_parser import OccupancyGridUtilities
from get_start_goal import get_start_goal
from comparison import astar_test, rrt_test, rrtstar_test

def print_menu():
    print("\nHello! Welcome to the comparison of rover path planning programs!")
    print("Type the corresponding number for which option you want to run to get started")
    print("1. A* vs RRT")
    print("2. A* vs RRT*")
    print("3. RRT vs RRT*")
    print("4. All three together")
    print("0. Exit")

def valid_grid_num(num_grids=12):
    while True:
        raw = input(f"Enter map number (0-{num_grids-1}): ").strip()
        try:
            grid_num = int(raw)
        except ValueError:
            print(f"'{raw}' isn't a valid number. Try again.")
            continue

        if grid_num < 0 or grid_num >= num_grids:
            print(f"Map number must be between 0 and {num_grids-1}. Try again.")
            continue

        return grid_num

def main():
    og_util = OccupancyGridUtilities()

    while True:
        print_menu()
        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            grid_num = valid_grid_num()
            grid = og_util.occupancy_grids[grid_num]
            start, goal = get_start_goal(og_util, grid_num)

            print("\nBeginning RRT analysis...")
            rrt_path, rrt_time = rrt_test(og_util, grid_num, start, goal)
            print(f"{rrt_time*1000:.2f}ms to run RRT")

            print("\nBeginning A* analysis...")
            astar_path, astar_time = astar_test(grid, start, goal)
            print(f"{astar_time*1000:.2f}ms to run A*")

            diff_ms = abs(astar_time - rrt_time) * 1000
            faster = "A*" if astar_time < rrt_time else "RRT"
            print(f"\nIn total, {faster} was faster by {diff_ms:.2f}ms!")
            
        elif choice == "2":
            grid_num = valid_grid_num()
            grid = og_util.occupancy_grids[grid_num]
            start, goal = get_start_goal(og_util, grid_num)

            print("\nBeginning RRT* analysis...")
            rrtstar_path, rrtstar_time = rrtstar_test(og_util, grid_num, start, goal)
            print(f"{rrtstar_time*1000:.2f}ms to run RRT*")

            print("\nBeginning A* analysis...")
            astar_path, astar_time = astar_test(grid, start, goal)
            print(f"{astar_time*1000:.2f}ms to run A*")

            diff_ms = abs(astar_time - rrtstar_time) * 1000
            faster = "A*" if astar_time < rrtstar_time else "RRT*"
            print(f"\nIn total, {faster} was faster by {diff_ms:.2f}ms!")
        elif choice == "3":
            grid_num = valid_grid_num()
            start, goal = get_start_goal(og_util, grid_num)

            print("\nBeginning RRT analysis...")
            rrt_path, rrt_time = rrt_test(og_util, grid_num, start, goal)
            print(f"{rrt_time*1000:.2f}ms to run RRT")

            print("\nBeginning RRT* analysis...")
            rrtstar_path, rrtstar_time = rrtstar_test(og_util, grid_num, start, goal)
            print(f"{rrtstar_time*1000:.2f}ms to run RRT*")

            diff_ms = abs(rrt_time - rrtstar_time) * 1000
            faster = "RRT" if rrt_time < rrtstar_time else "RRT*"
            print(f"\nIn total, {faster} was faster by {diff_ms:.2f}ms!")
        elif choice == "4":
            grid_num = valid_grid_num()
            grid = og_util.occupancy_grids[grid_num]
            start, goal = get_start_goal(og_util, grid_num)

            print("\nBeginning A* analysis...")
            astar_path, astar_time = astar_test(grid, start, goal)
            print(f"{astar_time*1000:.2f}ms to run A*")

            print("\nBeginning RRT analysis...")
            rrt_path, rrt_time = rrt_test(og_util, grid_num, start, goal)
            print(f"{rrt_time*1000:.2f}ms to run RRT")

            print("\nBeginning RRT* analysis...")
            rrtstar_path, rrtstar_time = rrtstar_test(og_util, grid_num, start, goal)
            print(f"{rrtstar_time*1000:.2f}ms to run RRT*")

            times = {"A*": astar_time, "RRT": rrt_time, "RRT*": rrtstar_time}
            fastest_name = min(times, key=times.get)
            slowest_name = max(times, key=times.get)
            diff_ms = abs(times[fastest_name] - times[slowest_name]) * 1000

            print(f"\nIn total, {fastest_name} was the fastest, "
                f"beating {slowest_name} by {diff_ms:.2f}ms!")
        else:
            print("Invalid choice, please try again.")





if __name__ == "__main__":
    main()