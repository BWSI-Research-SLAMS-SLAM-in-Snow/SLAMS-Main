#Essentially the same as ROR except you use a custom function and user defined weight to scale the radius
# based off of distance.
import math

'''TODO: Fix trigonometric calculations, 
        currently they are O(n^2) quadratic
        time complexity, which is not feasible for 
        an already complex multi-step pipeline.'''

class DROR:
    min_radius = 0
    min_neighbors = 0
    alpha = 0

    def __init__(self, min_radius, min_n, alpha):
        self.min_radius = min_radius
        self.min_neighbors = min_n
        self.alpha = alpha

    def filterCloudDyanamic(self, scan, angular_resolution):
        scan = scan.copy()
        n = len(scan)

        for i in range(n):
            if scan[i] == 0:
                continue
            rad = max(self.min_radius, self.alpha * scan[i] * angular_resolution)
            radius_squared = rad ** 2
            nCount = 0
            for j in range(n):
                if i == j or scan[j] == 0:
                    continue
                angle_diff = 2 * math.pi * (j - i) / n
                if angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                elif angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                r1 = scan[i]
                r2 = scan[j]
                distance_squared = (
                    r1**2 + r2**2
                    - 2 * r1 * r2 * math.cos(angle_diff)
                )
                if distance_squared <= radius_squared:
                    nCount += 1

            if nCount < self.min_neighbors:
                scan[i] = 0
                
        return scan



        

