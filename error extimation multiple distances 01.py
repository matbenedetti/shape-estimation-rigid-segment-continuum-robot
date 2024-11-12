import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# Define degrees
angles = []
error = []
second_angles = []  

# Read measured data
data01 = pd.read_csv("/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/Test135.csv")
data02 = pd.read_csv("/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/Test140.csv")
data03 = pd.read_csv("/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/Test145.csv")
df_trial01 = data01
df_trial02 = data02
df_trial03 = data03
df_trial = pd.DataFrame()

# Access 'Angle' as the index, not as a column
#degs = df_trial.index
degs = df_trial01['Angle']

for degr in df_trial01['Angle']:
    index = df_trial01.index[df_trial01['Angle'] == degr].tolist()
    if df_trial01[df_trial01['Angle'] == degr]['Angle'].values[0] <= -11:
        df_trial = pd.concat([df_trial, df_trial01.loc[index]])
        print(len(df_trial))
    elif df_trial01[df_trial01['Angle'] == degr]['Angle'].values[0] >= 12:
        df_trial = pd.concat([df_trial, df_trial03.loc[index]])
        print(len(df_trial))
    else:
        df_trial = pd.concat([df_trial, df_trial02.loc[index]])
        print(len(df_trial))


df_trial.reset_index(drop=True, inplace=True)

#print (df_trial)


# Extract 'y' and 'z' values corresponding to the filtered degrees
by_lst = df_trial['Bys_fft_100Hz']
bz_lst = df_trial['Bzs_fft_100Hz']
bx_lst = df_trial['Bxs_fft_100Hz']
phasediff = df_trial['Phase Difference']

folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/LUT/" 
file_name = "results35_imprvedcost.csv"
full_path = folder_path + file_name
files_database = ['LUT_30.csv', 'LUT_35.csv', 'LUT_40.csv', 'LUT_45.csv', 'LUT_50.csv', 'LUT_55.csv']




def get_angle( by, bz, bx, actual_angle, phasediff):
    min_cost_overall = float('inf')
    best_file = None
    best_angle = None
    i = 0
    for files in files_database:
        
        data = pd.read_csv(folder_path + files)
        cost_values = []
        lbnd = -35
        rbnd = 35
        angles = data['Angle']
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
        spline_y = CubicSpline(angles, y_vals_z)
        spline_z = CubicSpline(angles, z_vals_z)
        spline_x = CubicSpline(angles, x_vals_z)
        spline_phase = CubicSpline(angles, phase_z)
        spline_div = CubicSpline(angles, div_val_z)

        # Evaluate cost at different angles
        for index, angle in enumerate(angles):
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
            div_cost = abs(div_interp - div_z)

            # Total cost function
            cost =  angy_cost + angx_cost + angz_cost + phph
            cost_values.append(cost)

        # Get the angle with the minimum cost
        index = np.argmin(cost_values)
        min_angle = angles[index]

        phph = phase[phase.index[angles == 0].tolist()]
        phph = phph.values[-1]

        # Adjust min_angle based on phasediff (as per your custom logic)
        if min_angle > 0 and phasediff >= phph:
            while min_angle > 0:
                index = index - 1
                min_angle = angles[index]
        elif min_angle <= 0 and phasediff < phph:
            while min_angle < 0:
                index = index + 1
                min_angle = angles[index]

                
        # Get the angle with the minimum cost for the current file
        min_cost_for_file = np.min(cost_values)
        

        # Check if this is the smallest cost so far
        if min_cost_for_file < min_cost_overall:
            min_cost_overall = min_cost_for_file
            best_file = files
            best_angle = min_angle
    print(min_cost_overall)
    return best_angle, best_file




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



# Initialize variables to track the file with the lowest error statistics
lowest_mean_error = float('inf')
best_file = []
best_angles = []
best_error = []
best_wrong_angle = []


angles = []
errors = []
wrong_angles = []
files = []

for deg, by, bz, bx, ph in zip(degs, by_lst, bz_lst, bx_lst, phasediff):
    calc_angle, file = get_angle(by, bz, bx, deg, ph)
    angles.append(calc_angle)
    error = calc_angle - deg
    errors.append(error)
    if error > 0:
        wrong_angles.append(deg)
    best_file.append(file)
print('files used:', best_file)
# Error statistics
abs_errors = np.abs(errors)
mean_error, std_error, total_error = np.mean(abs_errors), np.std(abs_errors), np.sum(abs_errors)

print(f'Mean error: {mean_error}')
print(f'Standard deviation of error: {std_error}')
print(f'Total error: {total_error}')
print(f'Wrong angles: {wrong_angles}')

# Plot calculated vs actual angles and error distribution
plt.figure(figsize=(10, 6))
plt.plot(degs, angles, label='Calculated Angle', marker='o')
plt.plot(degs, degs, label='Actual Angle (y=x)', linestyle='--')
plt.title('Calculated Angle vs Actual Angle')
plt.xlabel('Actual Angle (Degrees)')
plt.ylabel('Calculated Angle')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(degs, errors, label='Error')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
plt.title('Error Distribution')
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()