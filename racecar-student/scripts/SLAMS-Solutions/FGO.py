'''This framework assumes that there is no loop closure being performed.'''
import numpy as np
class FactorGraphOptimizer:
    trajectory = []
    # Each element of edge with key i contains [t, z_i, omega_i]
    edges = {}
    def __init__(self, x0):
        self.trajectory.append(x0)
        self.trajectory.append(0)
    def update(self, newEdges):
        pass
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

        


