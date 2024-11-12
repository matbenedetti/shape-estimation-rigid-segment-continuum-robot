import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Assuming these values for weights and bounds
wt = 0.1  # Adjust as necessary
wn = 0.1  # Adjust as necessary
lbnd = 0
rbnd = 35
database = pd.read_csv('TOT_1100.csv', index_col=0)
filtered_database = database[database['Angle'].between(0, 35)]
# Example data (replace with your actual data)
angles = filtered_database['Angle']  # Database angles
y_vals = filtered_database['y']  # Corresponding y values from the database
z_vals = filtered_database['z']  # Corresponding z values from the database
bn = 0.000113926442935782# Example By_RMS value (replace with actual)
bt = 0.000418401637042686  # Example Bz_RMS value (replace with actual)

# Define the cost function
def minFunction(angle):
    angn_cost = wn * abs(np.interp(angle, angles, z_vals) - bn)
    angt_cost = wt * abs(np.interp(angle, angles, y_vals) - bt)
    cost = angt_cost + angn_cost
    return cost

# Generate a range of angles to evaluate the cost function
angle_range = np.linspace(lbnd, rbnd, 1000)
cost_values = [minFunction(angle) for angle in angle_range]

# Plot the cost function
plt.figure(figsize=(10, 6))
plt.plot(angle_range, cost_values, label="Cost Function")
plt.xlabel('Angle')
plt.ylabel('Cost')
plt.title('Cost Function vs. Angle')
plt.grid(True)
plt.legend()
plt.show()
