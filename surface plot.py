import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import warnings
from scipy.stats import mode


# Suppress ComplexWarning
#warnings.filterwarnings("ignore", category=np.ComplexWarning)


file_path = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_06112024/"
name_file = ['Testcoarse_30.csv']#, 'Test135.csv',  'Test140.csv', 'Test145.csv', 'Test150.csv', 'Test155.csv']
folder_path = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/LUT/"
label = ['30 mm', '35 mm', '40 mm', '45 mm', '50 mm', '55 mm']
files_database = ['LUT3_30.csv', 'LUT3_35.csv', 'LUT3_40.csv', 'LUT3_45.csv']
params = np.array([2.94648923, 2.21593223, 1.40937635, 0.76568571, 0.39475797])

gt_distance = 3.0

# Initialize lists to store matrices
y_matrix = []
z_matrix = []
x_matrix = []
angles = []
phase_files = []
tangents = []


single_row = np.random.normal(loc=5.89656935e-06 , scale=1.00000435e+00, size=(1, 144))

# Duplicate the row across all rows to create a consistent matrix
gaussian_matrix = np.tile(single_row, (15, 1))


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
distances = np.array([3.0, 3.5, 4.0, 4.5])

def plot_3d(matrix, angles, distances, axis_label):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    A, D = np.meshgrid(angles, distances)
    surf = ax.plot_surface(A, D, matrix, cmap='viridis')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Distance (mm)')
    ax.set_zlabel(f'{axis_label} Magnitude')
    fig.colorbar(surf)

# Original 3D plot for each axis
'''plot_3d(tangents, angles[0], distances, 'X Original')
plot_3d(y_matrix, angles[0], distances, 'Y Original')
plot_3d(z_matrix, angles[0], distances, 'Z Original')
plot_3d(phase_files, angles[0], distances, 'Phase Difference')'''

# Define interpolation range
coarser_angles = np.arange(-36, 36, 0.5)
coarser_distances = np.arange(3, 4.5, 0.1)


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

# Interpolated 3D plot for each axis
'''plot_3d(Z2_tan, coarser_angles, coarser_distances, 'X Interpolated')
plot_3d(Z2_y, coarser_angles, coarser_distances, 'Y Interpolated')
plot_3d(Z2_z, coarser_angles, coarser_distances, 'Z Interpolated')
plot_3d(Z2_phase, coarser_angles, coarser_distances, 'Phase Difference Interpolated')'''



# Load data from Test130.csv for overlay
test_data = pd.read_csv(file_path + name_file[0])
test_angles = test_data['Angle'].to_numpy()
test_y_vals = test_data['Bys_fft_100Hz'].to_numpy() 
test_z_vals = test_data['Bzs_fft_100Hz'].to_numpy()
test_x_vals = test_data['Bxs_fft_100Hz'].to_numpy()
test_phase_vals = test_data['Phase Difference'].to_numpy()
test_tan_vals = np.tan(test_phase_vals+0.16)


# Plotting the interpolated surfaces with Test130 components
fig = plt.figure()

# X Interpolated plot
ax1 = fig.add_subplot(231, projection='3d')
A, D = np.meshgrid(coarser_angles, coarser_distances)
surf_x = ax1.plot_surface(A, D, Z2_x, cmap='viridis', alpha=0.7)
ax1.scatter(test_angles, np.full_like(test_angles, gt_distance, dtype=float), (test_x_vals- x_mean)/x_std, color='r', label='Test37 X')
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Distance (mm)')
ax1.set_zlabel('X Magnitude')
ax1.legend()

# Y Interpolated plot
ax2 = fig.add_subplot(232, projection='3d')
surf_y = ax2.plot_surface(A, D, Z2_y, cmap='viridis', alpha=0.7)
ax2.scatter(test_angles, np.full_like(test_angles, gt_distance, dtype=float), (test_y_vals-y_mean)/y_std, color='r', label='Test37 Y')
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Distance (mm)')
ax2.set_zlabel('Y Magnitude')
ax2.legend()

# Z Interpolated plot
ax3 = fig.add_subplot(233, projection='3d')
surf_z = ax3.plot_surface(A, D, Z2_z, cmap='viridis', alpha=0.7)
ax3.scatter(test_angles, np.full_like(test_angles, gt_distance, dtype=float), (test_z_vals- z_mean)/z_std, color='r', label='Test37 Z')
ax3.set_xlabel('Angle (degrees)')
ax3.set_ylabel('Distance (mm)')
ax3.set_zlabel('Z Magnitude')
ax3.legend()

# Phase Difference Interpolated plot
ax4 = fig.add_subplot(234, projection='3d')
surf_phase = ax4.plot_surface(A, D, Z2_phase, cmap='viridis', alpha=0.7)
ax4.scatter(test_angles, np.full_like(test_angles, gt_distance, dtype=float), (test_phase_vals - phase_mean)/phase_std, color='r', label='Test37 Phase Difference')
ax4.set_xlabel('Angle (degrees)')
ax4.set_ylabel('Distance (mm)')
ax4.set_zlabel('Phase Difference')
ax4.legend()

# Phase Difference Interpolated plot
ax5 = fig.add_subplot(235, projection='3d')
surf_phase = ax5.plot_surface(A, D, Z2_tan, cmap='viridis', alpha=0.7)
ax5.scatter(test_angles, np.full_like(test_angles, gt_distance, dtype=float), (test_tan_vals - tan_mean)/tan_std, color='r', label='Test37 Tangent')
ax5.set_xlabel('Angle (degrees)')
ax5.set_ylabel('Distance (mm)')
ax5.set_zlabel('Tangent')
ax5.legend()

plt.tight_layout()
plt.show()



def find_closest_angle_distance(magnitude, Z2, coarser_angles, coarser_distances, mean, std):
    # Calculate the absolute difference between x_magnitude and  magnitude values in Z2_x
    magnitude = (magnitude - mean)/ std
    difference = np.abs(Z2 - magnitude)
    
    return difference

angles = []
distances = []
bx_values = []
by_values = []
bz_values = []
phase_values = []
tan_values = []
error = []

for file_idx, file in enumerate(name_file):
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
        differencey = find_closest_angle_distance(bz, Z2_z, coarser_angles, coarser_distances, z_mean, z_std)
        differencez = find_closest_angle_distance(by, Z2_y, coarser_angles, coarser_distances, y_mean, y_std)
        differencephase = find_closest_angle_distance(ph, Z2_phase, coarser_angles, coarser_distances, phase_mean, phase_std)
        differencetan = find_closest_angle_distance(tan, Z2_tan, coarser_angles, coarser_distances, tan_mean, tan_std)

        #difference = differencex  + differencephase + differencey +   differencez + differencetan
        # Variables for the differences
        differences = np.array([differencex, differencephase, differencey, differencez, differencetan])

        # Calculate 'difference' using dot product
        difference = np.dot(params, differences)


        # Find the index of the minimum difference
        min_index = np.unravel_index(np.argmin(difference), difference.shape)
        # Retrieve corresponding angle and distance
        best_distance = coarser_distances[min_index[0]]
        best_angle = coarser_angles[min_index[1]]
        # Find the option with the minimum difference

        '''plot_3d(difference, coarser_angles, coarser_distances, f'for angle {angle}')
        #plot a vertical line at best angle and best distance
        plt.plot([best_angle, best_angle], [best_distance, best_distance], [0, 200], color='red', linewidth=2, label="Best Angle & Distance")
        plt.plot([angle, angle], [gt_distance, gt_distance], [0, 200], color='black', linewidth=2, label="actual Angle & Distance")

        plt.show()'''

        


        # Append the best angle, distance, and corresponding Bx, By, Bz, and phase values
        angles.append(best_angle)
        distances.append(best_distance)
        bx_values.append(bx)
        by_values.append(by)
        bz_values.append(bz)
        phase_values.append(ph)
        tan_values.append(tan)
        error.append(angle - best_angle)


    

# Convert lists to arrays if needed
angles = np.array(angles)
distances = np.array(distances)
bx_values = np.array(bx_values)
by_values = np.array(by_values)
bz_values = np.array(bz_values)
phase_values = np.array(phase_values)
tan_values = np.array(tan_values)
error = np.array(error)


# Plot the values
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(degs, angles, 'o-', label="Best Angle ")
plt.plot(degs, degs, '--', label="y=x", color='black')
plt.xlabel('Actual Angle (Degrees)')
plt.ylabel('Calculated Angle (Degrees)')
plt.title("Best Angle vs Actual Angle")
plt.grid(True)

#arrays of 3 with length of degs
array = [gt_distance]*len(degs)
plt.subplot(2, 1, 2)
plt.plot(degs, distances, 'o-', label="Calculated distance")
plt.plot(degs, array , '--', label="Ground truth", color='black')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Calculated Distance (mm)')
plt.title("Best Distance over angles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(Z2_x.shape)  # Should match (number of coarser_distances, number of coarser_angles)
print(Z2_y.shape)
print(Z2_z.shape)
print(Z2_phase.shape)
print(np.sqrt(np.mean(error**2)))



# Create a figure for 3D plotting
fig = plt.figure(figsize=(15, 10))

# Generate meshgrid for angles and distances
A, D = np.meshgrid(coarser_angles, coarser_distances)
  

# 3D plot for bx values with interpolated surface
ax1 = fig.add_subplot(321, projection='3d')
surf_x = ax1.plot_surface(A, D, Z2_x, cmap='viridis', alpha=0.7)
ax1.scatter(angles, distances, (bx_values-x_mean)/x_std, color='blue', marker='o', label='Test Data Bx')
ax1.set_xlabel('Angle (degrees)')
ax1.set_ylabel('Distance (mm)')
ax1.set_zlabel('Bx Magnitude')
ax1.legend()

# 3D plot for by values with interpolated surface
ax2 = fig.add_subplot(322, projection='3d')
surf_y = ax2.plot_surface(A, D, Z2_y, cmap='viridis', alpha=0.7)
ax2.scatter(angles, distances, (by_values - y_mean)/y_std, color='green', marker='o', label='Test Data By')
ax2.set_xlabel('Angle (degrees)')
ax2.set_ylabel('Distance (mm)')
ax2.set_zlabel('By Magnitude')
ax2.legend()

# 3D plot for bz values with interpolated surface
ax3 = fig.add_subplot(323, projection='3d')
surf_z = ax3.plot_surface(A, D, Z2_z, cmap='viridis', alpha=0.7)
ax3.scatter(angles, distances, (bz_values-z_mean)/z_std, color='red', marker='o', label='Test Data Bz')
ax3.set_xlabel('Angle (degrees)')
ax3.set_ylabel('Distance (mm)')
ax3.set_zlabel('Bz Magnitude')
ax3.legend()

# 3D plot for phase difference with interpolated surface
ax4 = fig.add_subplot(324, projection='3d')
surf_phase = ax4.plot_surface(A, D, Z2_phase, cmap='viridis', alpha=0.7)
ax4.scatter(angles, distances, (phase_values-phase_mean)/phase_std, color='purple', marker='o', label='Test Data Phase Difference')
ax4.set_xlabel('Angle (degrees)')
ax4.set_ylabel('Distance (mm)')
ax4.set_zlabel('Phase Difference')
ax4.legend()

ax5 = fig.add_subplot(325, projection='3d')
surf_phasetan = ax5.plot_surface(A, D, Z2_tan, cmap='viridis', alpha=0.7)
ax5.scatter(angles, distances, (tan_values-tan_mean)/tan_std, color='purple', marker='o', label='Test Tan Phase Difference')
ax5.set_xlabel('Angle (degrees)')
ax5.set_ylabel('Distance (mm)')
ax5.set_zlabel('Phase Difference')
ax5.legend()

# Add color bars to each subplot
fig.colorbar(surf_x, ax=ax1, shrink=0.5, aspect=5)
fig.colorbar(surf_y, ax=ax2, shrink=0.5, aspect=5)
fig.colorbar(surf_z, ax=ax3, shrink=0.5, aspect=5)
fig.colorbar(surf_phase, ax=ax4, shrink=0.5, aspect=5)
fig.colorbar(surf_phasetan, ax=ax5, shrink=0.5, aspect=5)


plt.tight_layout()
plt.show()