import math
class ROR:
    def __init__(self, radius, min_n):
        self.radius = radius
        self.min_neighbors = min_n

    def filterCloud(self, scan):
        scan = scan.copy()
        radius_squared = self.radius ** 2
        n = len(scan)

        for i in range(n):
            if scan[i] == 0:
                continue
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



        

