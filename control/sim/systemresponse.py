import os
import pandas as pd
import numpy as np
from pid_sm import PIDController
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

# Function to simulate the system response using the PID controller
@lru_cache(maxsize=None)  # None means the cache can grow without bound
def simulate_system_response(kp, ki, kd, setpoint, object_centers):
    pid = PIDController(kP=kp, kI=ki, kD=kd)
    pid.reset()

    data = {'Input': [], 'kP': [], 'kI': [], 'kD': [], 'Output': []}

    for object_center in object_centers:
        error = setpoint - object_center
        output = pid.update(error)

        # Append values to the data dictionary
        data['Input'].append(error)
        data['kP'].append(kp)
        data['kI'].append(ki)
        data['kD'].append(kd)
        data['Output'].append(output)

    return pd.DataFrame(data)

# Generate input values
object_centers = np.arange(0, 641, 10)

# Setpoint for all cases
setpoint = 320

# Get user input for kp, ki, and kd values
kp = float(input("Enter the value for kp: "))
ki = float(input("Enter the value for ki: "))
kd = float(input("Enter the value for kd:"))

# Convert numpy.ndarray to tuple for cacheability
object_centers_tuple = tuple(object_centers)

# Specify the folder path
folder_path = 'control/sim/outputData/'  # Adjust the relative path as needed

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Specify the filename for the combined results
combined_filename = os.path.join(folder_path, 'combined_system_response_data.xlsx')

# Check if the file already exists
if os.path.exists(combined_filename):
    # Load existing data from the file
    existing_df = pd.read_excel(combined_filename)

    # Append the new data to the existing DataFrame
    new_df = simulate_system_response(kp, ki, kd, setpoint, object_centers_tuple)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

else:
    # Create a new DataFrame if the file doesn't exist
    combined_df = simulate_system_response(kp, ki, kd, setpoint, object_centers_tuple)

# Print the head of the combined DataFrame
print("\nHead of the Combined DataFrame:")
print(combined_df.head())

# Save the combined DataFrame to the file
combined_df.to_excel(combined_filename, index=False)

print(f'Data saved to {combined_filename}')
