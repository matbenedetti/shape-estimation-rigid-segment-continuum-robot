import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
fft_df = pd.read_csv('/local/home/matildebenedetti/Downloads/Tesi_new/04_Shape estimation/data_14102024/Trial0_31mm.csv')

# Filter for the trial 0 data
fft_df = fft_df[fft_df['Trial'] == 0]

# Convert angles to radians
angles = np.radians(fft_df['Angle'])

# Extract magnetic field components
Bx = fft_df['Bxs_fft_100Hz']
By = fft_df['Bys_fft_100Hz']
Bz = fft_df['Bzs_fft_100Hz']

# Radius of the arc
r = 1

# Parametric equations for the arc (half circumference)
x_values = r * np.cos(angles)
y_values = r * np.sin(angles)

# Prepare 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot vectors along the arc
for i in range(len(angles)):
    # Starting point of each vector (on the arc)
    start_x = x_values[i]
    start_y = y_values[i]
    start_z = 0  # All vectors start at z=0
    
    # Plot the vector using Bx, By, Bz components
    ax.quiver(start_x, start_y, start_z, Bx[i], By[i], Bz[i], color='b')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits
ax.set_xlim([-1.5, 1.5])  # Adjust X limits based on arc radius
ax.set_ylim([-1.5, 1.5])  # Adjust Y limits based on arc radius
ax.set_zlim([np.min(Bz) - 1, np.max(Bz) + 1])  # Adjust Z limits based on Bz values

# Optional: set equal aspect ratio for better visualization
ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1

# Show the plot
plt.show()
