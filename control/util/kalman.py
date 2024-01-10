import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Create the filter
my_filter = KalmanFilter(dim_x=2, dim_z=1)

# Initialize the filter's matrices
my_filter.x = np.array([[0.], [0.]])  # initial state (position and velocity)
my_filter.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
my_filter.H = np.array([[1., 0.]])  # Measurement function
my_filter.P *= 1000.  # covariance matrix
my_filter.R = 5  # measurement uncertainty
my_filter.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.1)  # process uncertainty

# Simulate detected results
detected_results = range(0, 641, 100)

for measurement in detected_results:
    my_filter.predict()
    my_filter.update(np.array([[measurement]]))

    # Access the smoothed output
    smoothed_output = my_filter.x[0, 0]

    # Do something with the smoothed output (e.g., control your motors)
    print(f"Measurement: {measurement}, Smoothed Output: {smoothed_output}")
