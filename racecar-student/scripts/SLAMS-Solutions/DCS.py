'''Flow: have a constant flow of encoder and imu fused position, 
   and use a huber loss function to detect when the LiDAR data diverges.
   When this happens, trigger DCS. Scale the LiDAR covariance and allow the
   EKF to reconverge until stability is reached. '''