import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('CalibrationDC8.csv', index_col='Index')

# Extract the columns as numpy arrays and convert to microtesla
Bxs = df['x'].values * 1e6
Bys = df['y'].values * 1e6
Bzs = df['z'].values * 1e6

# Define a function to select two points and filter the data
def select_points_and_filter(B_values, label):
    fig, ax = plt.subplots()
    ax.plot(B_values, label=f'{label} Data')
    ax.set_title(f'Select two points on the {label} plot')
    plt.xlabel('Index')
    plt.ylabel(f'{label} (µT)')
    plt.legend()
    
    # Capture two points
    points = plt.ginput(2)
    plt.close(fig)
    
    # Convert the x-coordinates of the selected points to integer indices
    index1, index2 = int(points[0][0]), int(points[1][0])
    
    # Make sure index1 is less than index2
    if index1 > index2:
        index1, index2 = index2, index1
    
    # Filter the data between the selected points
    filtered_data = B_values[index1:index2+1]
    
    return filtered_data, index1, index2

# Select and filter Bx data
Bx_filtered, Bx_start, Bx_end = select_points_and_filter(Bxs, 'Bx')

# Select and filter By data
By_filtered, By_start, By_end = select_points_and_filter(Bys, 'By')

# Select and filter Bz data
Bz_filtered, Bz_start, Bz_end = select_points_and_filter(Bzs, 'Bz')

# Plot the filtered data
Bx_midpoint_idx = Bx_start + len(Bx_filtered) // 2
By_midpoint_idx = By_start + len(By_filtered) // 2
Bz_midpoint_idx = Bz_start + len(Bz_filtered) // 2

best_mean_value = np.mean([Bx_midpoint_idx, By_midpoint_idx, Bz_midpoint_idx])

# Plot the filtered data and mark the midpoint
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot filtered Bx data
axs[0].plot(np.arange(Bx_start, Bx_end + 1), Bx_filtered, label='Filtered Bx')
axs[0].plot(Bx_midpoint_idx, Bx_filtered[len(Bx_filtered) // 2], 'ro', label='Midpoint')
axs[0].set_title('Filtered Bx Data')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Bx (µT)')
axs[0].legend()

# Plot filtered By data
axs[1].plot(np.arange(By_start, By_end + 1), By_filtered, label='Filtered By')
axs[1].plot(By_midpoint_idx, By_filtered[len(By_filtered) // 2], 'ro', label='Midpoint')
axs[1].set_title('Filtered By Data')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('By (µT)')
axs[1].legend()

# Plot filtered Bz data
axs[2].plot(np.arange(Bz_start, Bz_end + 1), Bz_filtered, label='Filtered Bz')
axs[2].plot(Bz_midpoint_idx, Bz_filtered[len(Bz_filtered) // 2], 'ro', label='Midpoint')
axs[2].set_title('Filtered Bz Data')
axs[2].set_xlabel('Index')
axs[2].set_ylabel('Bz (µT)')
axs[2].legend()

plt.tight_layout()
plt.show()
