import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings


# File paths and initializations
file_path_1 = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_06112024/"
file_path_2 = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_08112024/"
name_file = ['Testcoars_40.csv', 'Testcoars_30.csv', 'Testcoars_35.csv', 'Testcoars_32.csv', 'Testcoars_37.csv','Testcoars_42.csv', 'Testcoars_45.csv']
folder_path = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/LUT/"
files_database = ['LUT3_30.csv', 'LUT3_35.csv', 'LUT3_40.csv', 'LUT3_45.csv']
distances = np.array([3.0, 3.5, 4.0, 4.5])

# Initialize lists to store matrices
y_matrix, z_matrix, x_matrix, angles, phase_files, tangents, fun, mu_lst, sigma_lst = [], [], [], [], [], [], [], [], []

def plot_3d(matrix, angles, distances, axis_label):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    A, D = np.meshgrid(angles, distances)
    surf = ax.plot_surface(A, D, matrix, cmap='viridis')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Distance (mm)')
    ax.set_zlabel(f'{axis_label} Magnitude')
    fig.colorbar(surf)

# Load data from CSV files
for file in files_database:
    data = pd.read_csv(folder_path + file)
    angles_data = data['Angle']
    y_matrix.append(data['Bys_fft_100Hz'])
    z_matrix.append(data['Bzs_fft_100Hz'])
    x_matrix.append(data['Bxs_fft_100Hz'])
    phase_files.append(data['Phase Difference'])
    tangents.append(np.tan(data['Phase Difference'] + 0.16))
    angles.append(angles_data)

# Convert lists to numpy arrays for further operations
y_matrix = np.array(y_matrix)
z_matrix = np.array(z_matrix)
x_matrix = np.array(x_matrix)
phase_files = np.array(phase_files)
tangents = np.array(tangents)
angles = np.array(angles)

# Normalization of matrices
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

# Define interpolation range
coarser_angles = np.arange(-36, 36, 0.5)
coarser_distances = np.arange(3, 4.5, 0.1)

# Interpolate X, Y, Z, phase, and tan matrices
interp_spline_x = RectBivariateSpline(distances, angles[0], x_matrix)
interp_spline_y = RectBivariateSpline(distances, angles[0], y_matrix)
interp_spline_z = RectBivariateSpline(distances, angles[0], z_matrix)
interp_spline_phase = RectBivariateSpline(distances, angles[0], phase_files)
interp_spline_tan = RectBivariateSpline(distances, angles[0], tangents)

Z2_x = interp_spline_x(coarser_distances, coarser_angles)
Z2_y = interp_spline_y(coarser_distances, coarser_angles)
Z2_z = interp_spline_z(coarser_distances, coarser_angles)
Z2_phase = interp_spline_phase(coarser_distances, coarser_angles)
Z2_tan = interp_spline_tan(coarser_distances, coarser_angles)


# Define helper function for closest angle and distance
def find_closest_angle_distance(magnitude, Z2, mean, std):
    normalized_magnitude = (magnitude - mean) / std
    return np.abs(Z2 - normalized_magnitude)

def gaussian(x, mu, sigma):
    return 1-((1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2))


# Define optimization objective function
# Define optimization objective function
def objective_function(params):
    A, B, C, D, E =  [1,1,1,1,1]

    total_mse = 0
    mu, sigma = params
    print(mu, sigma)
    # Generate a single row with normally distributed values
    single_row = gaussian(coarser_angles, mu, sigma)

    # Duplicate the row across all rows to create a consistent matrix
    gaussian_matrix = np.tile(single_row, (15, 1))


    for index, file in enumerate(name_file):
        if index == 0:
            file_path = file_path_2
        else:
            file_path = file_path_1
        test_data = pd.read_csv(file_path + file)
        error = []

        # Load test data and pre-compute tangent values
        test_angles = test_data['Angle'].to_numpy()
        test_y_vals = test_data['Bys_fft_100Hz'].to_numpy()
        test_z_vals = test_data['Bzs_fft_100Hz'].to_numpy()
        test_x_vals = test_data['Bxs_fft_100Hz'].to_numpy()
        test_phase_vals = test_data['Phase Difference'].to_numpy()
        test_tan_vals = np.tan(test_phase_vals + 0.16)
        
        # Loop over each data point in the test file
        for angle, by, bz, bx, ph, tan in zip(test_angles, test_y_vals, test_z_vals, test_x_vals, test_phase_vals, test_tan_vals):
            differencex = find_closest_angle_distance(bx, Z2_x, x_mean, x_std)
            differencez = find_closest_angle_distance(bz, Z2_z, z_mean, z_std)
            differencey = find_closest_angle_distance(by, Z2_y, y_mean, y_std)
            differencephase = find_closest_angle_distance(ph, Z2_phase, phase_mean, phase_std)
            differencetan = find_closest_angle_distance(tan, Z2_tan, tan_mean, tan_std)
            
            # Weighted sum of differences
            difference = A * differencex +   gaussian_matrix * differencephase + C * differencez + D * differencey + E * differencetan
            if angle ==5:
                plot_3d( differencephase, coarser_angles, coarser_distances, f'gaussian_matrix * differencephase')
            '''plot_3d(differencephase, coarser_angles, coarser_distances, f'differencephase')'''


            plt.show()
            # Find the best-matching angle
            best_angle = coarser_angles[np.unravel_index(np.argmin(difference), difference.shape)[1]]
            error.append(angle - best_angle)

        # Calculate mean squared error for the file
        total_mse += np.sqrt(np.mean(np.square(error)))
    #print(total_mse / len(name_file))
    # Average error across all files
    fun.append(total_mse / len(name_file))
    mu_lst.append(mu)
    sigma_lst.append(sigma)

    return total_mse / len(name_file)


# Run minimization
#initial_guess = [1, 1, 1, 1, 1]
initial_guess = [0, 10]
# Define options for the Nelder-Mead method
options = {
    'initial_simplex': None,  # None uses the default initialization; you can also define your own
    'alpha': 6.0,  # The reflection coefficient (default is 1.0, higher values encourage bigger steps)
    'gamma': 2.0,  # The expansion coefficient (default is 2.0, higher values lead to bigger expansions)
    'rho': 0.2,  # The contraction coefficient (default is 0.5, smaller values encourage larger steps)
    'sigma': 0.2  # The shrinkage coefficient (default is 0.5, smaller values lead to larger steps)
}

bounds = [(None, None), (1e-6, None)]  # Lower bound on initial_guess[1] to ensure positivity
result = minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds, options=options)
#result = minimize(objective_function, initial_guess, method='Nelder-Mead')
params = result.x
if result.success:
    print("Optimal parameters (A, B, C, D, E):", result.x)
    print("Minimum error:", result.fun)
else:
    print("Optimization failed:", result.message)

print(result)



# Plot the optimized Gaussian
single_row = gaussian(coarser_angles, params[0], params[1])  
plt.plot(coarser_angles, single_row)
plt.title("Optimized Gaussian")
plt.xlabel("Angle")
plt.ylabel("Weight")
plt.show()

gaussian_matrix = np.tile(single_row, (15, 1))
plot_3d(gaussian_matrix, coarser_angles, coarser_distances, f'Gaussian')

plt.show()


plt.subplot(3,1,1)
plt.plot(range(len(fun)), fun)
plt.xlabel('Iteration')
plt.ylabel('Error(degrees)')
plt.title('Error vs Iteration')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(range(len(mu_lst)), mu_lst)
plt.xlabel('Iteration')
plt.ylabel('mu')
plt.title('mu vs Iteration')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(range(len(sigma_lst)), sigma_lst)
plt.xlabel('Iteration')
plt.ylabel('sigma')
plt.title('sigma vs Iteration')
plt.grid(True)
plt.show()