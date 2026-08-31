'''Flow: have a tuned threshold for how many iterations an obstacle needs to hold constant.
   Run this persistence checker over the grid and lower the probability of cells that are higher
   but fell under this threshold.'''
#TODO: implement set probability function for occupancy grid class.
import numpy as np
class OccupancyGridVerifier:
    
    def __init__(self, width, height, min_iterations=3, occupancy_threshold=0.7):
        self.width = width
        self.height = height
        self.min_iterations = min_iterations
        self.occupancy_threshold = occupancy_threshold
        self.persistence = np.zeros((width, height), dtype=np.int32)

    def update(self, occupancy_grid):
        for x in range(self.width):
            for y in range(self.height):
                probability = occupancy_grid.get_odds(
                    x * occupancy_grid.resolution,
                    y * occupancy_grid.resolution
                )
                if probability >= self.occupancy_threshold:
                    self.persistence[x, y] += 1
                else:
                    self.persistence[x, y] = 0
        return self.persistence >= self.min_iterations
    
    def is_verified(self, x, y):
        return self.persistence[x, y] >= self.min_iterations

    def reset(self):
        self.persistence.fill(0)

    def filtered_grid(self, occupancy_grid):
        for x in range(self.width):
            for y in range(self.height):
                probability = occupancy_grid.get_odds(
                   x * occupancy_grid.resolution,
                   y * occupancy_grid.resolution
                )
                if self.persistence[x][y] == False:
                    probability = 0.5
                occupancy_grid.set(x,y,probability)
                
      