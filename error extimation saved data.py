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
data = pd.read_csv("/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_22102024/Trial35bssx.csv")
df_trial = data
# Access 'Angle' as the index, not as a column
#degs = df_trial.index
degs = df_trial['Angle']


# Extract 'y' and 'z' values corresponding to the filtered degrees
by_lst = df_trial['Bys_fft_100Hz']
bz_lst = df_trial['Bzs_fft_100Hz']
bx_lst = df_trial['Bxs_fft_100Hz']
phasediff = df_trial['Phase Difference']

folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/LUT/" 
file_name = "results35_imprvedcost.csv"
full_path = folder_path + file_name



def get_angle(data, by, bz, bx, actual_angle, phasediff):
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
        cost =    angx_cost  + div_cost + angy_cost + angz_cost + phph
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

    #plot_min_function(angles, cost_values, lbnd, rbnd, min_angle, actual_angle)

    return min_angle




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


database = pd.read_csv(folder_path + 'LUT_35.csv')

wrong_angle = []



# Iterate over each degree to calculate the error
for deg, by, bz, bx, ph in zip(degs, by_lst, bz_lst, bx_lst, phasediff):
    # Load the database for angle calculation
    calculated_angle = get_angle(database, by, bz, bx, deg, ph)    
    # Store results
    angles.append(calculated_angle)
    error.append(calculated_angle - deg)
    if error[-1] > 0:
        wrong_angle.append(deg)
abs_error = [abs(e) for e in error] 
total_error = sum(abs_error)
#mean and standard deviation of the error
mean_error = np.mean(abs_error)
std_error = np.std(abs_error)
print('mean error:', mean_error)
print('std error:', std_error)
print('tot error:', total_error)
print('wrong_angle:', wrong_angle)



# Create DataFrame with matching lengths
df_error = pd.DataFrame({
    'angle': degs,
    'c_angle': angles,
    'error': error,
})

# Save the results to CSV
df_error.to_csv(full_path, index=True)
 



y_min = min(by_lst.min(), bz_lst.min(), bx_lst.min(), (database['Bys_fft_100Hz']).min(), 
            (database['Bzs_fft_100Hz']).min(), (database['Bxs_fft_100Hz']).min())
y_max = max(by_lst.max(), bz_lst.max(), bx_lst.max(), (database['Bys_fft_100Hz']).max(), 
            (database['Bzs_fft_100Hz']).max(), (database['Bxs_fft_100Hz']).max())
y_min = y_min - 0.1 * abs(y_min)
y_max = y_max + 0.1 * abs(y_max)

plt.figure(figsize=(14, 16))

# Plot adjusted By data
plt.subplot(3, 1, 1)
plt.plot(degs, by_lst, marker='o', linestyle='-', color='b', label='By to be estimated')
plt.plot(database['Angle'], database['Bys_fft_100Hz'], marker='x', linestyle='--', color='g', label='By (Database)')
# Adding red dots for wrong angles
#by_wrong_values = [by_lst[np.where(np.array(degs) == angle)[0][0]] for angle in wrong_angle]
#plt.scatter(wrong_angle, by_wrong_values, color='r', marker='o', s=100, label='Wrong Angle')
plt.xlabel('Degrees')
plt.ylabel('By')
plt.title('By vs Degrees')
plt.legend()
plt.grid(True)
plt.ylim(y_min, y_max)  # Set common y-axis limits

# Plot adjusted Bz data
plt.subplot(3, 1, 2)
plt.plot(degs, bz_lst, marker='o', linestyle='-', color='b', label='Bz to be estimated')
plt.plot(database['Angle'], database['Bzs_fft_100Hz'], marker='x', linestyle='--', color='m', label='Bz (Database)')
# Adding red dots for wrong angles
'''bz_wrong_values = [bz_lst[np.where(np.array(degs) == angle)[0][0]] for angle in wrong_angle]
plt.scatter(wrong_angle, bz_wrong_values, color='r', marker='o', s=100, label='Wrong Angle')'''
plt.xlabel('Degrees')
plt.ylabel('Bz')
plt.title('Bz vs Degrees')
plt.legend()
plt.grid(True)
plt.ylim(y_min, y_max)  # Set common y-axis limits

# Plot adjusted Bx data
plt.subplot(3, 1, 3)
plt.plot(degs, bx_lst, marker='o', linestyle='-', color='b', label='Bx to be estimated')
plt.plot(database['Angle'], database['Bxs_fft_100Hz'], marker='x', linestyle='--', color='g', label='Bx (Database)')
# Adding red dots for wrong angles
'''bx_wrong_values = [bx_lst[np.where(np.array(degs) == angle)[0][0]] for angle in wrong_angle]
plt.scatter(wrong_angle, bx_wrong_values, color='r', marker='o', s=100, label='Wrong Angle')'''
plt.xlabel('Degrees')
plt.ylabel('Bx')
plt.title('Bx vs Degrees')
plt.legend()
plt.grid(True)
plt.ylim(y_min, y_max)  # Set common y-axis limits

plt.tight_layout()
plt.show()





# Plotting actual angle vs calculated angle and error in a separate figure
plt.figure(figsize=(10, 8))

# Plot actual angle vs calculated angle
plt.subplot(2, 1, 1)
plt.plot(degs, angles, marker='o', linestyle='-', color='b', label='Calculated Angle')
plt.plot(degs, degs, marker='x', linestyle='--', color='r', label='Actual Angle')
plt.xlabel('Degrees')
plt.ylabel('Angle')
plt.title('Actual vs Calculated Angle')
plt.legend()
plt.grid(True)

# Plot error
plt.subplot(2, 1, 2)
plt.plot(degs, error, marker='o', linestyle='-', color='b', label='Error')
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.title('Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure()
plt.plot(degs, phasediff)
plt.show()