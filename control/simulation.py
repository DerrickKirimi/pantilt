import numpy as np
import matplotlib.pyplot as plt

from pid import PIDController  

# Function to simulate the system response using the PID controller
def simulate_system_response(p, i, d, setpoint, object_centers):
    angles = []

    for object_center in object_centers:
        pid = PIDController(p, i, d)
        pid.reset()

        error = setpoint - object_center
        output = pid.update(error)
        angles.append(output)

    return angles

# Generate input values
object_centers = np.arange(0, 641, 10)

# PID controller parameters
p = 1.5
i = 0.1
d = 0

# Setpoint for all cases
setpoint = 320

# Simulate system response for all object center values
angles_all = simulate_system_response(p, i, d, setpoint, object_centers)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(object_centers, angles_all, label='System Response')
plt.title('PID Controller Output vs. Object Center')
plt.xlabel('Object Center')
plt.ylabel('PID Controller Output (Angle)')
plt.legend()
plt.grid(True)
plt.show()
