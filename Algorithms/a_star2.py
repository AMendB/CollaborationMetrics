import numpy as np
import heapq

import numpy as np
import heapq

def heuristic(a, b):
    """Heuristic function: Manhattan Distance"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_find_path(mapa, start, goal, return_distance=False):
    """A* algorithm to find the shortest path on a map."""
    # Possible directions: up, down, left, right and diagonals
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),
               (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # Initialize the priority queue with the start node
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Dictionary to store the accumulated cost from the start node
    g_score = {start: 0}
    
    # Dictionary to store the path
    came_from = {}
    
    while open_set:
        # Pop the node with the lowest estimated cost
        current = heapq.heappop(open_set)[1]
        
        # If we reached the goal, reconstruct the path and return the distance
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            if return_distance:
                return len(path) - 1  # The distance is the number of steps
            else:
                return path
        
        # Explore neighbors
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check if the neighbor is within the map and navigable
            if (0 <= neighbor[0] < mapa.shape[0] and 
                0 <= neighbor[1] < mapa.shape[1] and 
                mapa[neighbor[0], neighbor[1]] == 1):
                
                # Calculate the accumulated cost to the neighbor
                tentative_g_score = g_score[current] + 1
                
                # If the neighbor has not been visited or we found a better path
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    
    # If no path is found, return -1
    return -1

# Example usage
mapa = np.array([
    [1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1]
])

if __name__ == '__main__':
    scenario_map = np.genfromtxt(f'Environment/Maps/challenging_map_big.csv', delimiter=',')
    start = (10, 10)
    goal = (62, 13)

    distance = a_star_find_path(scenario_map, start, goal, return_distance=True)
    print(f"Distance between {start} and {goal}: {distance}")

    path = a_star_find_path(scenario_map, start, goal)

    # Visualize the map and the found path
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(scenario_map, cmap='binary')

    if path:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], 'b-', linewidth=3, label='Path')
        plt.plot(path[0, 1], path[0, 0], 'go', markersize=15, label='Start')
        plt.plot(path[-1, 1], path[-1, 0], 'ro', markersize=15, label='Goal')

    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("A* Pathfinding Result")
    plt.show()