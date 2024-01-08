#1
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
folder_path = 'control/sim/outputData/'  

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Create a string to append to the file name
param_string = f"_kp_{kp}_ki_{ki}_kd_{kd}"

# Using ProcessPoolExecutor for parallel execution
with ProcessPoolExecutor() as executor:
    # Simulate system response for user-specified PID parameters
    futures = [executor.submit(simulate_system_response, kp, ki, kd, setpoint, object_centers_tuple)]
    df_list = [future.result() for future in futures]

# Concatenate DataFrames from different processes
df = pd.concat(df_list, ignore_index=True)

# Print the head of the DataFrame
print("\nHead of the DataFrame:")
print(df.head())
print(df.tail())

# Save the DataFrame to an Excel file in the specified folder
excel_filename = os.path.join(folder_path, f'system_response_data{param_string}.xlsx')
df.to_excel(excel_filename, index=False)

print(f'Data saved to {excel_filename}')
