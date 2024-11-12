import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# Define degrees
angles = []
error = []
second_angles = []  

# Read measured data
data = pd.read_csv("C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_09102024/Tial8_35mm.csv")
df_trial = data
# Access 'Angle' as the index, not as a column
#degs = df_trial.index
degs = df_trial['Angle']


# Extract 'y' and 'z' values corresponding to the filtered degrees
by_lst = df_trial['Bys_fft_100Hz']
bz_lst = df_trial['Bzs_fft_100Hz']
bx_lst = df_trial['Bxs_fft_100Hz']
phasediff = df_trial['Phase Difference']

folder_path = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_09102024/" 
file_name = "results35_imprvedcost.csv"
full_path = folder_path + file_name








def plot_min_function(angle_range, cost_values, lbnd, rbnd, min_angle, actual_angle):
                  
    # Plot the cost function
    plt.figure(figsize=(8, 6))
    plt.plot(angle_range, cost_values, label='Cost Function', color='b')
    
    # Mark the minimum angle with a vertical line
    plt.axvline(min_angle, color='r', linestyle='--', label=f'Min Angle: {min_angle:.2f}°')
    plt.axvline(actual_angle, color='g', linestyle='--', label=f'Actual Angle: {actual_angle:.2f}°')
    # Title with actual angle
    plt.title(f'Minimization of the Cost Function (Actual Angle: {actual_angle:.2f}°)')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.show()

files_database = ['LUT_30_mm.csv', 'LUT_35_mm.csv', 'LUT_40_mm.csv']


wrong_angle = []

def get_side(phase_diff):
    if phase_diff + np.pi/2 < 0:
        return 'right'
    else:
        return 'left'
    

def get_angle(angles_est, folder_path, ph):
    cost_values = []
    for angle in angles_est:
        data = pd.read_csv(folder_path + angle[1])
        phase = data['Phase Difference']
        angle_range = data['Angle']
        ph_spline = CubicSpline(angle_range, phase)
        ph_interp = ph_spline(angle[0])
        phph = abs(ph_interp - ph)
        cost_values.append((phph, angle[0], angle[1]))
    # take the minimum of the firs column of cost_values
    min_cost = min(cost_values, key=lambda x: x[0])
    min_angle = min_cost[1]
    distance = min_cost[2]
    return min_angle, distance

def angles_at_distances(files_database,folder_path, bz, bx, side):
    angles_est = []

    for files in files_database:
        cost_values = []

        data = pd.read_csv(folder_path + files)
        if side == 'left':
            data = data[data['Angle'] <= 0]
        else:
            data = data[data['Angle'] > 0]
            #reset index
            data = data.reset_index(drop=True)

        angles = data['Angle']
        z_vals = data['Bzs_fft_100Hz']
        x_vals = data['Bxs_fft_100Hz']
        phase = data['Phase Difference']

        div = x_vals / z_vals

        div_z = (bx/bz)
        spline_div = CubicSpline(angles, div)
        spline_ph = CubicSpline(angles, phase)

        # Evaluate cost at different angles
        for index, angle in enumerate(angles):
            div_interp = spline_div(angle)
            ph_interp = spline_ph(angle)

            # Compute costs using normalized values
            div_cost = abs(div_interp - div_z)
            ph_cost = abs(ph_interp - ph)

            # Total cost function
            cost = div_cost
            cost_values.append(cost)
        # Get the angle with the minimum cost
        index = np.argmin(cost_values)
        min_angle = angles[index]
        angles_est.append([min_angle, files])

        
    return angles_est

files_data = []
        
    
# Iterate over each degree to calculate the error
for deg, by, bz, bx, ph in zip(degs, by_lst, bz_lst, bx_lst, phasediff):
    # Load the database for angle calculation
    side = get_side(ph)
    angles_est = angles_at_distances(files_database,folder_path, bz, bx, side)
    calculated_angle, distance = get_angle(angles_est, folder_path, ph)    
    # Store results
    files_data.append(distance)
    angles.append(calculated_angle)
    error.append(calculated_angle - deg)
    if error[-1] > 0:
        wrong_angle.append(deg)
abs_error = [abs(e) for e in error] 
total_error = sum(abs_error)
print(files_data)






# Convert lists to numpy arrays for easier handling
angles = np.array(angles)
degs = np.array(degs)
error = np.array(error)
abs_error = np.array(abs_error)

# Plot 1: Calculated Angle vs. Measured Angle
plt.figure(figsize=(8, 6))
plt.plot(degs, angles, label='Calculated Angles', marker='o', color='b')
plt.plot(degs, degs, label='Measured Angles (y=x line)', linestyle='--', color='g')
plt.xlabel('Measured Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.title('Calculated Angle vs. Measured Angle')
plt.grid(True)
plt.legend()
plt.show()

import matplotlib.cm as cm
import numpy as np

# Generate a color map to differentiate the files
unique_files = list(set(files_data))  # Get the unique files
color_map = cm.get_cmap('viridis', len(unique_files))  # Choose a color map (viridis is one option)

# Create a mapping between files and colors
file_to_color = {file: color_map(i) for i, file in enumerate(unique_files)}

# Plot 1: Calculated Angle vs. Measured Angle with different colors for different files
plt.figure(figsize=(8, 6))

# Plot each point with the corresponding color based on file_data
for deg, angle, file in zip(degs, angles, files_data):
    plt.scatter(deg, angle, color=file_to_color[file], label=f'{file}', s=100)

# Plot the y=x line for reference
plt.plot(degs, degs, label='Measured Angles (y=x line)', linestyle='--', color='g')

plt.xlabel('Measured Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.title('Calculated Angle vs. Measured Angle (Colored by File)')
plt.grid(True)

# To avoid duplicate labels, ensure only unique labels appear in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()


import matplotlib.cm as cm
import numpy as np

# Determine left or right classification for each angle
sides = [get_side(ph) for ph in phasediff]

# Create a color map for 'left' and 'right'
colors = {'left': 'blue', 'right': 'red'}  # Blue for left, Red for right

# Plot 2: Calculated Angle vs. Measured Angle with different colors for 'left' and 'right'
plt.figure(figsize=(8, 6))

# Plot each point with the corresponding color based on side
for deg, angle, side in zip(degs, angles, sides):
    plt.scatter(deg, angle, color=colors[side], label=side, s=100)

# Plot the y=x line for reference
plt.plot(degs, degs, label='Measured Angles (y=x line)', linestyle='--', color='g')

plt.xlabel('Measured Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.title('Calculated Angle vs. Measured Angle (Colored by Side: Left or Right)')
plt.grid(True)

# To avoid duplicate labels, ensure only unique labels appear in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
