import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings

# Suppress ComplexWarning
warnings.filterwarnings("ignore", category=np.ComplexWarning)


# Read a file and extract the data
folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/"
file_name = "Cubic Fitting Tan 0.csv"
label = ['30','35','40', '45', '50', '55']
csv_files = ["Test030.csv",  "Test035.csv", "Test040.csv", "Test045.csv", "Test050.csv", "Test055.csv"]
full_path = folder_path + file_name
files_database = ['LUT_30.csv', 'LUT_35.csv', 'LUT_40.csv', 'LUT_45.csv', 'LUT_50.csv', 'LUT_55.csv']
folder_path_data = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/LUT/"

df = pd.read_csv("/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_22102024/" + file_name)
a_list = df['a']
b_list = df['b']
c_list = df['c']
d_list = df['d']
c_angle_files = []  

# Set up the figure for combined plotting of angle comparison and error
plt.figure(figsize=(10, 12))
phase_files = []


def get_angle(by, bz, bx, angles, phasediff):
    min_cost_overall = float('inf')
    best_file = None
    best_angle = None
    i = 0
    for file_index, files in enumerate(files_database):
        angle = angles[file_index]
        data = pd.read_csv(folder_path_data + files)
        cost_values = []
        angles_data = data['Angle']
        y_vals = data['Bys_fft_100Hz']
        z_vals = data['Bzs_fft_100Hz']
        x_vals = data['Bxs_fft_100Hz']
        phase = data['Phase Difference']
        div = x_vals / z_vals


        # Normalize using Z-scores
        y_mean, y_std = np.mean(y_vals), np.std(y_vals)
        z_mean, z_std = np.mean(z_vals), np.std(z_vals)
        x_mean, x_std = np.mean(x_vals), np.std(x_vals)
        phase_mean, phase_std = np.mean(phase), np.std(phase)
        div_mean, div_std = np.mean(div), np.std(div)

        # Apply Z-score normalization
        y_vals_z = (y_vals - y_mean) / y_std
        z_vals_z = (z_vals - z_mean) / z_std
        x_vals_z = (x_vals - x_mean) / x_std
        phase_z = (phase - phase_mean) / phase_std
        div_val_z = (div -div_mean) / div_std

        by_z = (by - y_mean) / y_std
        bz_z = (bz - z_mean) / z_std
        bx_z = (bx - x_mean) / x_std
        phasediff_z = (phasediff - phase_mean) / phase_std
        div_z = ((bx/bz)-div_mean)/div_std

        # Spline interpolation for better handling of flat regions
        spline_y = CubicSpline(angles_data, y_vals_z, extrapolate= True)
        spline_z = CubicSpline(angles_data, z_vals_z, extrapolate= True)
        spline_x = CubicSpline(angles_data, x_vals_z, extrapolate= True)
        spline_phase = CubicSpline(angles_data, phase_z, extrapolate= True)
        spline_div = CubicSpline(angles_data, div_val_z, extrapolate= True)
        # Evaluate cost at different angles
        y_interp = spline_y(angle)
        z_interp = spline_z(angle)
        x_interp = spline_x(angle)
        ph_interp = spline_phase(angle)
        div_interp = spline_div(angle)

        # Compute costs using normalized values
        angy_cost = abs(y_interp - by_z)
        angz_cost = abs(z_interp - bz_z)
        angx_cost = abs(x_interp - bx_z)
        phph = abs(ph_interp - phasediff_z)

        # Total cost function
        cost =   phph + angx_cost + angz_cost + angy_cost
        cost_values.append(cost)
        # Get the angle with the minimum cost
        index = np.argmin(cost_values)

                
        # Get the angle with the minimum cost for the current file
        min_cost_for_file = np.min(cost_values)
        

        # Check if this is the smallest cost so far
        if min_cost_for_file < min_cost_overall:
            min_cost_overall = min_cost_for_file
            best_file = files
            best_angle = angle

    return best_angle, best_file



for file_idx, file in enumerate(csv_files):

    data = pd.read_csv(folder_path + file)
    df_trial = data

    degs = df_trial['Angle']
    phase = []
    error = []
    angles = []
    errors = []
    files_used = []

    
    # Extract 'y', 'z', and 'x' values corresponding to the filtered degrees
    by_lst = df_trial['Bys_fft_100Hz']
    bz_lst = df_trial['Bzs_fft_100Hz']
    bx_lst = df_trial['Bxs_fft_100Hz']
    phasediff = df_trial['Phase Difference']

    for angle, by, bz, bx, ph in zip(degs, by_lst, bz_lst, bx_lst, phasediff):
        c_angle = []

        for i in range(len (a_list)):
            a = a_list[i]
            b = b_list[i]
            c = c_list[i]
            d = d_list[i]
                
            ph = np.tan(phasediff[angle+35] + 0.16)
            # Calculate the error
            coefficients = [a, b, c, d - ph]
            estim_angle = np.roots(coefficients)
            
            # Take the angle that is between -35 and 35
            found_angle = None
            for root in estim_angle:
                if np.isreal(root):  # Check if the root is real
                    if -35 < root < 35:
                        found_angle = root
                        break
            if found_angle is None:
                estim_angle = [np.real(angle) for angle in estim_angle]
                found_angle = min(estim_angle, key=lambda x: min(abs(x + 35), abs(x - 35)))
            
            c_angle.append(found_angle)
            
            # Save error
            error.append(abs(found_angle - angle))

        print('angle:', len(c_angle))

        calculated_angle, best_file = get_angle(by, bz, bx, c_angle, ph)    
        # Store results
        files_used.append(best_file) 
        angles.append(calculated_angle)
        errors.append(calculated_angle - angle)
        phase.append(ph)

        # Plot actual angle vs calculated angle
    print('files used:', files_used)
    plt.subplot(2, 1, 1)
    plt.plot(degs, angles, marker='o', linestyle='-', label=f'Calculated Angle ({label[file_idx]})')
    plt.plot(degs, degs, marker='x', linestyle='--', color='r', label='Actual Angle' if file_idx == 0 else "")
    plt.xlabel('Degrees')
    plt.ylabel('Angle')
    plt.title('Actual vs Calculated Angle')
    plt.legend()
    plt.grid(True)

    # Plot error vs angle
    plt.subplot(2, 1, 2)
    plt.plot(degs, errors, marker='o', linestyle='-', label=f'Error ({label[file_idx]})')
    plt.xlabel('Degrees')
    plt.ylabel('Error')
    plt.title('Error vs Angle')
    plt.legend()
    plt.grid(True)


    phase_files.append(phase)
    c_angle_files.append(angles)
    
    fitted_line = a * degs**3 + b * degs**2 + c * degs + d

plt.tight_layout()
plt.show()

# Create a new figure for the phase difference and fitted line
plt.figure(figsize=(10, 6))
for file_idx, file in enumerate(csv_files):
    plt.plot(degs, phase_files[file_idx], label=f'Phase Difference ({label[file_idx]})')
for i in range(len (a_list)):
    a = a_list[i]
    b = b_list[i]
    c = c_list[i]
    d = d_list[i]
    fitted_line = a * degs**3 + b * degs**2 + c * degs + d

    # Plot the phase difference and fitted line on the same figure
    plt.plot(degs, fitted_line, label=f'Fitted Line')

plt.xlabel('Degrees')
plt.ylabel('Phase / Fitted Line')
plt.title('Phase Difference and Fitted Line')
plt.legend()
plt.grid(True)

# Display both figures
plt.tight_layout()
plt.show()
