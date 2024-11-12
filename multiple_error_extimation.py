import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# Read measured data

folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/"
csv_files = ["Test030.csv"]#,  "Test135.csv","Test037.csv", "Test140.csv", "Test142.csv", "Test145.csv", "Test150.csv", "Test155.csv"]
label = ['30','35', '37','40', '42', '45', '50', '55']


    
file_name = 'Error_35mm.csv'
full_path = folder_path + file_name


def get_angle(data, by, bz, bx, actual_angle, phasediff):
    cost_values = []
    angles = data['Angle']
    y_vals = data['Bys_fft_100Hz']
    z_vals = data['Bzs_fft_100Hz']
    x_vals = data['Bxs_fft_100Hz']
    phase = data['Phase Difference']

    # Normalize using Z-scores
    y_mean, y_std = np.mean(y_vals), np.std(y_vals)
    z_mean, z_std = np.mean(z_vals), np.std(z_vals)
    x_mean, x_std = np.mean(x_vals), np.std(x_vals)
    phase_mean, phase_std = np.mean(phase), np.std(phase)

    # Apply Z-score normalization
    y_vals_z = (y_vals - y_mean) / y_std
    z_vals_z = (z_vals - z_mean) / z_std
    x_vals_z = (x_vals - x_mean) / x_std
    phase_z = (phase - phase_mean) / phase_std

    by_z = (by - y_mean) / y_std
    bz_z = (bz - z_mean) / z_std
    bx_z = (bx - x_mean) / x_std
    phasediff_z = (phasediff - phase_mean) / phase_std

    # Spline interpolation for better handling of flat regions
    spline_y = CubicSpline(angles, y_vals_z)
    spline_z = CubicSpline(angles, z_vals_z)
    spline_x = CubicSpline(angles, x_vals_z)
    spline_phase = CubicSpline(angles, phase_z)

    # Evaluate cost at different angles
    for index, angle in enumerate(angles):
        y_interp = spline_y(angle)
        z_interp = spline_z(angle)
        x_interp = spline_x(angle)
        ph_interp = spline_phase(angle)

        # Compute costs using normalized values
        angy_cost = abs(y_interp - by_z)
        angz_cost = abs(z_interp - bz_z)
        angx_cost = abs(x_interp - bx_z)
        phph = abs(ph_interp - phasediff_z)

        # Total cost function
        cost =  angx_cost  + phph + angz_cost + angy_cost
        cost_values.append(cost)

    # Get the angle with the minimum cost
    index = np.argmin(cost_values)
    min_angle = angles[index]

    # Adjust min_angle based on phasediff (as per your custom logic)
    phph = phase[phase.index[angles == 0].tolist()]
    phph = phph.values[-1]

    if min_angle > 0 and phasediff >= phph:
        while min_angle > 0 and index > 0:
            index -= 1
            min_angle = angles[index]
    elif min_angle <= 0 and phasediff < phph:
        while min_angle < 0 and index < len(angles) - 1:
            index += 1
            min_angle = angles[index]

    return min_angle


# Load the database
database = pd.read_csv('/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/LUT/LUT_30.csv')

# Initialize an empty list to store DataFrames
all_results = []

# Iterate over each file
for file_idx, file in enumerate(csv_files):
    # Read the file
    df_trial = pd.read_csv(folder_path + file)
    degs = df_trial['Angle']

    # Extract 'y', 'z', 'x', and 'phase difference' values
    by_lst = df_trial['Bys_fft_100Hz']
    bz_lst = df_trial['Bzs_fft_100Hz']
    bx_lst = df_trial['Bxs_fft_100Hz']
    phasediff = df_trial['Phase Difference']

    angles = []
    error = []
    wrong_angle = []

    # Iterate over each degree to calculate the error
    for deg, by, bz, bx, ph in zip(degs, by_lst, bz_lst, bx_lst, phasediff):
        # Load the database for angle calculation
        calculated_angle = get_angle(database, by, bz, bx, deg, ph)
        angles.append(calculated_angle)
        error.append(abs(calculated_angle - deg))
        if error[-1] > 0:
            wrong_angle.append(deg)

    # Create DataFrame for this file's results
    df_error = pd.DataFrame({
        'angle': degs,
        'c_angle': angles,
        'error': error,
        'Trial': file_idx * np.array([1] * len(degs))
    })

    # Append this file's DataFrame to the list
    all_results.append(df_error)

# Concatenate all DataFrames
final_df = pd.concat(all_results, ignore_index=True)

# Save the results to CSV
final_df.to_csv(full_path, index=False)

print("Data saved successfully to", full_path)



# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# (Your existing code for data processing goes here...)

# Plot c_angle vs angle for every trial
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)

plt.plot(df_error['angle'], df_error['angle'], label='y=x', color='black', linestyle='--')  # Add y=x line

# Iterate over the trials and plot c_angle vs angle

for file_idx, df_error in enumerate(all_results):
    plt.subplot(2, 1, 1)

    plt.plot(df_error['angle'], df_error['c_angle'], label=f'Trial {label[file_idx]}', marker='o')
    plt.subplot(2, 1, 2)
    plt.plot(df_error['angle'], df_error['error'], marker='o', linestyle='-', label='Error')

# Customize the plot
plt.subplot(2, 1, 1)

plt.title('Calculated Angle vs Original Angle for All Trials')
plt.xlabel('Original Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.xlabel('Degrees')
plt.ylabel('Error')
plt.title('Error')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

