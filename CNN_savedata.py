import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import warnings
from scipy.stats import mode
import seaborn as sns


file_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_06112024/"
name_file = ['Test_30.csv', 'Test_32.csv',  'Test_35.csv', 'Test_37.csv', 'Test_40.csv', 'Test_42.csv']
folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/LUT/"
label = ['30 mm', '35 mm', '40 mm', '45 mm', '50 mm', '55 mm']
files_database = ['LUT2_30.csv', 'LUT2_35.csv', 'LUT2_40.csv', 'LUT2_45.csv', 'LUT2_50.csv', 'LUT2_55.csv']
gt_distance = [3.0, 3.2, 3.5, 3.7, 4.0, 4.2]

# Initialize lists to store matrices
y_matrix = []
z_matrix = []
x_matrix = []
angles = []
phase_files = []
tangents = []

# Define a function to save heatmaps
def save_heatmap(matrix, coarser_angles, coarser_distances, axis_label, output_dir, angle, dist):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=np.round(coarser_angles, 2),
        yticklabels=np.round(coarser_distances, 2),
        cmap='viridis',
        cbar=True
    )
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Distance (mm)')
    plt.title(f'{axis_label} Heatmap')
    file_path = os.path.join(output_dir, f'{axis_label}_{dist}_{angle}_heatmap.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to avoid memory leaks


def find_closest_angle_distance(magnitude, Z2, coarser_angles, coarser_distances, mean, std):
    # Calculate the absolute difference between x_magnitude and  magnitude values in Z2_x
    magnitude = (magnitude - mean)/ std
    difference = np.abs(Z2 - magnitude)
    
    return difference

# Load data from CSV files
for file_index, files in enumerate(files_database):
    data = pd.read_csv(folder_path + files)
    angles_data = data['Angle']
    y_vals = data['Bys_fft_100Hz']
    z_vals = data['Bzs_fft_100Hz']
    x_vals = data['Bxs_fft_100Hz']
    ph = data['Phase Difference']
    tan = np.tan(ph+0.16)
    
    # Append the values for each test
    angles.append(angles_data)
    y_matrix.append(y_vals)
    z_matrix.append(z_vals)
    x_matrix.append(x_vals)
    phase_files.append(ph)
    tangents.append(tan)

# Convert to NumPy arrays for easier manipulation
y_matrix = np.array(y_matrix)
z_matrix = np.array(z_matrix)
x_matrix = np.array(x_matrix)
phase_files = np.array(phase_files)
tangents = np.array(tangents)



#normilize the data
y_mean, y_std = np.mean(y_matrix), np.std(y_matrix)
z_mean, z_std = np.mean(z_matrix), np.std(z_matrix)
x_mean, x_std = np.mean(x_matrix), np.std(x_matrix)
phase_mean, phase_std = np.mean(phase_files), np.std(phase_files)
tan_mean, tan_std = np.mean(tangents), np.std(tangents)

y_matrix = (y_matrix - y_mean) / y_std
z_matrix = (z_matrix - z_mean) / z_std
x_matrix = (x_matrix - x_mean) / x_std
phase_files = (phase_files - phase_mean) / phase_std
tangents = (tangents - tan_mean) / tan_std


angles = np.array(angles)

# Set up distances (6 distances from label)
distances = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5])



# Define interpolation range
coarser_angles = np.arange(-36, 36, 0.5)
coarser_distances = np.arange(3, 5.5, 0.1)

# Interpolation for X, Y, and Z matrices
interp_spline_x = RectBivariateSpline(distances, angles[0], x_matrix)
interp_spline_y = RectBivariateSpline(distances, angles[0], y_matrix)
interp_spline_z = RectBivariateSpline(distances, angles[0], z_matrix)
interp_spline_phase = RectBivariateSpline(distances, angles[0], phase_files)
interp_spline_tan = RectBivariateSpline(distances, angles[0], tangents)
# Generate interpolated data for each axis
Z2_x = interp_spline_x(coarser_distances, coarser_angles)
Z2_y = interp_spline_y(coarser_distances, coarser_angles)
Z2_z = interp_spline_z(coarser_distances, coarser_angles)
Z2_phase = interp_spline_phase(coarser_distances, coarser_angles)
Z2_tan = interp_spline_tan(coarser_distances, coarser_angles)

# Create a directory to store heatmaps if it doesn't already exist
output_dir = "heatmap_data"
os.makedirs(output_dir, exist_ok=True)

labels = []

for file in name_file:
    test_data = pd.read_csv(file_path + file)
    test_angles = test_data['Angle'].to_numpy()
    test_y_vals = test_data['Bys_fft_100Hz'].to_numpy() 
    test_z_vals = test_data['Bzs_fft_100Hz'].to_numpy()
    test_x_vals = test_data['Bxs_fft_100Hz'].to_numpy()
    test_phase_vals = test_data['Phase Difference'].to_numpy()
    test_tan_vals = np.tan(test_phase_vals+0.16)


    angles = []
    distances = []
    bx_values = []
    by_values = []
    bz_values = []
    phase_values = []
    tan_values = []
    error = []

    for file_idx, file in enumerate(name_file):
        dist = gt_distance[file_idx]
        data = pd.read_csv(file_path + file)
        df_trial = data

        degs = df_trial['Angle']

        # Extract 'y', 'z', and 'x' values corresponding to the filtered degrees
        by_lst = df_trial['Bys_fft_100Hz']
        bz_lst = df_trial['Bzs_fft_100Hz']
        bx_lst = df_trial['Bxs_fft_100Hz']
        phasediff = df_trial['Phase Difference']
        tan_lst = np.tan(phasediff+0.16)

        for angle, by, bz, bx, ph, tan in zip(degs, by_lst, bz_lst, bx_lst, phasediff, tan_lst):
            differencex = find_closest_angle_distance(bx, Z2_x, coarser_angles, coarser_distances, x_mean, x_std)
            differencez = find_closest_angle_distance(bz, Z2_z, coarser_angles, coarser_distances, z_mean, z_std)
            differencey = find_closest_angle_distance(by, Z2_y, coarser_angles, coarser_distances, y_mean, y_std)
            differencephase = find_closest_angle_distance(ph, Z2_phase, coarser_angles, coarser_distances, phase_mean, phase_std)
            differencetan = find_closest_angle_distance(tan, Z2_tan, coarser_angles, coarser_distances, tan_mean, tan_std)
            
            
            # Save heatmaps for all interpolated data
            save_heatmap(differencex, coarser_angles, coarser_distances, 'X_diff', output_dir, angle, dist)
            save_heatmap(differencey, coarser_angles, coarser_distances, 'Y_diff', output_dir, angle, dist)
            save_heatmap(differencez, coarser_angles, coarser_distances, 'Z_diff', output_dir, angle, dist)
            save_heatmap(differencephase, coarser_angles, coarser_distances, 'Phase_diff', output_dir, angle, dist)
            save_heatmap(differencetan, coarser_angles, coarser_distances, 'Tangent_diff', output_dir, angle, dist)


            # Create a DataFrame with tuples
            labels.append((angle, dist))        

            # Append the best angle, distance, and corresponding Bx, By, Bz, and phase values

            bx_values.append(bx)
            by_values.append(by)
            bz_values.append(bz)
            phase_values.append(ph)
            tan_values.append(tan)


    

# Convert lists to arrays if needed
angles = np.array(angles)
distances = np.array(distances)
bx_values = np.array(bx_values)
by_values = np.array(by_values)
bz_values = np.array(bz_values)
phase_values = np.array(phase_values)
tan_values = np.array(tan_values)
error = np.array(error)

