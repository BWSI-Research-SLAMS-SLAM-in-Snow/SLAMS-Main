'''Flow: have a constant flow of encoder and imu fused position, 
   and use a huber loss function to detect when the LiDAR data diverges.
   When this happens, trigger DCS. Scale the LiDAR covariance and allow the
   EKF to reconverge until stability is reached. '''

import numpy as np
class DynamicCovarianceScheduling:
    def __init__(self, alpha, threshold, max_scale=100):
        self.alpha = alpha
        self.threshold = threshold     
        self.max_scale = max_scale

    def huber_loss(self, residual):
        r = abs(residual)
        if r <= self.alpha:
            return 0.5 * r * r
        else:
            return self.alpha * (r - 0.5 * self.alpha)

    def DCS(self, loss):
        if loss <= self.threshold:
            return 1.0
        scale = 1.0 + (loss - self.threshold)
        return min(scale, self.max_scale)

    def update(self, lidar_pose, imu_encoder_pose, lidar_covariance):
        residual = np.linalg.norm(lidar_pose - imu_encoder_pose)
        loss = self.huber_loss(residual)
        scale = self.DCS(loss)
        scaled_covariance = scale * lidar_covariance
        return scaled_covariance, loss, scale
        
              
      
        