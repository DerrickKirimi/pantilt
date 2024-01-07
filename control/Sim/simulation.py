#2
import os
import pandas as pd
import numpy as np
from pid_sm import PIDController

# Function to simulate the system response using the PID controller
def simulate_system_response(pid_controller, setpoint, object_centers):
    data = {'Input': [], 'kP': [], 'kI': [], 'kD': [], 'Output': []}

    for object_center in object_centers:
        error = setpoint - object_center
        output = pid_controller.update(error)

        # Append values to the data dictionary
        data['Input'].append(error)
        data['kP'].append(pid_controller.kP)
        data['kI'].append(pid_controller.kI)
        data['kD'].append(pid_controller.kD)
        data['Output'].append(output)

    return pd.DataFrame(data)

# Generate input values
object_centers = np.arange(0, 641, 10)

# Setpoint for all cases
setpoint = 320

# Get user input for kp, ki, and kd values
kp = float(input("Enter the value for kp: "))
ki = float(input("Enter the value for ki: "))
kd = float(input("Enter the value for kd: "))

# Initialize the PID controller with user-specified parameters
pid = PIDController(kP=kp, kI=ki, kD=kd)
pid.reset()

# Simulate system response for all object center values
df = simulate_system_response(pid, setpoint, object_centers)

# Specify the folder path
folder_path = '../TuningData/'  # Adjust the relative path as needed

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Create a string to append to the file name
param_string = f"_kp_{kp}_ki_{ki}_kd_{kd}"

# Save the DataFrame to an Excel file in the specified folder
excel_filename = os.path.join(folder_path, f'system_response_data{param_string}.xlsx')
df.to_excel(excel_filename, index=False)

print("\nHead of the DataFrame:")
print(df.head())

print(f'Data saved to {excel_filename}')
