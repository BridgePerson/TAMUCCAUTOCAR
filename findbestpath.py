from matplotlib import pyplot as plt
import cv2
import numpy as np
from heapq import heappush, heappop

def asearchalgorithm(self, grid):
    shortest_path = a_star_graph_search(
            start               = (0,0), #starting location
            goal_function       = get_goal_function(grid), #location wanted
            successor_function  = get_successor_function(grid), #better pathing 
            heuristic           = get_heuristic(grid)  #finding the math needed for the pathing       
        )
    if shortest_path is None or grid[0][0] == 1:
        return -1
    else:
        return len(shortest_path)

def a_star_graph_search(start, goal_function, successor_function, heuristic):
    visited = set() #adds coordinates to visited set.
    came_from= dict() #where the robot came from.
    distance = {start: 0} #distance from start.
    frontier = PriorityQueue() #priorityqueue where everything is stored.
    frontier.add(start)
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        if goal_function(node):
            return reconstruct_path(came_from, start, node)
        visited.add(node)
        for successor in successor_function(node):
            frontier.add(successor, priority = distance[node] + 1 + heuristic(successor)
                         )
            if (successor not in distance or distance[node] + 1 < distance[successor]):
                distance[successor] = distance[node] +1
                came_from[successor] = node
    return None

def reconstruct_path(came_from, start, end):
    """
    >>> came_from = {'b': 'a', 'c': 'a', 'd': 'c', 'e': 'd', 'f': 'd'}
    >>> reconstruct_path(came_from, 'a', 'e')
    ['a', 'c', 'd', 'e']
    """
    reverse_path = [end]
    while end != start:
        end = came_from[end]
        reverse_path.append(end)
    return list(reversed(reverse_path))

#Goal function sees whether we have reached the final goal of the path
def get_goal_function(grid):
    """
    >>> f = get_goal_function([[0, 0], [0, 0]])
    >>> f((0, 0))
    False
    >>> f((0, 1))
    False
    >>> f((1, 1))
    True
    """
    M = len(grid)
    N = len(grid[0])
    def is_bottom_right(cell):
        return cell == (M-1, N-1)
    return is_bottom_right

#Sucessor function
#Function is to find the cells adjacent to the current cell
def get_successor_function(grid):
    """
    >>> f = get_successor_function([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> sorted(f((1, 2)))
    [(0, 1), (0, 2), (2, 1), (2, 2)]
    >>> sorted(f((2, 1)))
    [(1, 0), (1, 2), (2, 2)]
    """
    def get_clear_adjacent_cells(cell):
        i,j = cell
        return (
            (i + a, j + b)
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            if a != 0 or b != 0
            if 0 <= i + a < len(grid)
            if 0 <= j + b < len(grid[0])
            if grid[i+a][j + b] == 0
        )
    return get_clear_adjacent_cells

#Heuristic
#The goal of the heuristic is to find the distance to the goal in a clear grid of the same size.
def get_heuristic(grid):
    """
    >>> f = get_heuristic([[0, 0], [0, 0]])
    >>> f((0, 0))
    1
    >>> f((0, 1))
    1
    >>> f((1, 1))
    0
    """
    M, N = len(grid)
    (a, b) = goal_cell = (M - 1, N - 1)
    def get_clear_path_distance_from_goal(cell):
        (i, j) = cell
        return max(abs(a - i), abs(b-j))
    return get_clear_path_distance_from_goal

#Priority queue
class PriorityQueue:

    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))

    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))

    def pop(self):
        priority, value = heappop(self.heap)
        return value

    def __len__(self):
        return len(self.heap)
    

def plottingCV2(x1,y1,x2,y2):
    #  Draw a Rectangle
    earth = np.zeros((800,800,3), dtype='uint8')
    #cv2.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
    cv2.circle(earth, (x1, y1), 5, (0,0,255), thickness=-1)
    cv2.circle(earth, (x2, y2), 5, (0,0,255), thickness=-1)
    cv2.line(earth, (x1,y1), (x2,y2), (0,0,255), thickness=1)
    cv2.imshow('Path', earth)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def plottingPLT(x1,y1,x2,y2):
    plt.scatter(x1,y1)
    plt.scatter(x2,y2)
    x=[x1, x2]
    y=[y1, y2]
    plt.plot(x,y)
    c=plt.imshow
    plt.show()


def main():
    plottingCV2(100,220,700, 720)
    plottingPLT(100,220,700, 720)
    


if __name__ == "__main__":
    main()
