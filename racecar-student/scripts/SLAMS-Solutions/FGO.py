'''This framework assumes that there is no loop closure being performed.'''
import numpy as np
class FactorGraphOptimizer:
    trajectory = []
    # Each element of edge with key i contains [t, z_i, omega_i y]
    # y == 0 means that the current factor edge is an odometry estimate, while 1 is ICP
    # y helps determine which jacobian to choose
    edges = {}
    num_edges = 0
    x_t = 0
    def __init__(self, x0):
        self.trajectory.append(x0)
        self.trajectory.append(0)
    def update(self, z_icp, z_imu, omega_icp, omega_imu):
        #determine delta x, a vector of increments that is made
        #to the trajectory. 
        num_edges += 1
        self.edges[num_edges] = np.array([self.x_t, z_icp, omega_icp, 1]) 
        num_edges += 1
        self.edges[num_edges] = np.array([self.x_t, z_imu, omega_imu, 0]) 
        self.x_t += 1
                
    def edgeCost(self, edge):
        nextNode = self.edges[edge]
        residual = self.trajectory[nextNode[0]+1] - (self.trajectory[nextNode[0]] + nextNode[1])
        cost = residual @ nextNode[2] @ residual.T
        return cost
    def trajectoryCost(self):
        cost = 0
        for edge in self.edges:
            cost += self.edgeCost(self, edge)
        return cost    
    def get_trajectory(self):
        return self.trajectory

        


