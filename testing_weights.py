import numpy as np
import pandas as pd
import serial
from scipy import optimize
import matplotlib.pyplot as plt
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe

# Initialize variables
degs = np.arange(0, 36, 1)
COM = "COM3"
BAUD = 115200
no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500
data_list = []
by_lst = []
bz_lst = []
error = []
angles = []
error_lst = []

# Function to calculate angle based on data and input bn, bt values
def get_angle(data, bz, by):
    lbnd = 0
    rbnd = 35
    angles = data['Angle']
    print(angles)
    y_vals = data['y']
    z_vals = data['z']
    wz = 0.01  # Weight for y (adjust to optimize)
    wy = 0.01  # Weight for z (adjust to optimize)
    
    def minFunction(angle):
        angn_cost = wz * abs(np.interp(angle, angles, z_vals) - bz)
        angt_cost = wy * abs(np.interp(angle, angles, y_vals) - by)
        cost = angt_cost + angn_cost
        return cost
    
    result = optimize.fminbound(minFunction, lbnd, rbnd, full_output=1)
    xopt, fval, ierr, numfunc = result

    # Print the summary
    print("Summary of fminbound results:")
    print(f"Optimal value of x (minimizer): {xopt}")
    print(f"Function value at the minimizer: {fval}")
    print(f"Error flag (0 if converged): {ierr}")
    print(f"Number of function evaluations: {numfunc}")
    return xopt

# Load measured data
df_measured = pd.read_csv('measured_data.csv')

# Loop over each measured By_RMS and Bz_RMS to calculate angles and errors
database = pd.read_csv('TOT_1100.csv', index_col=0)
filtered_database = database[database['Angle'].between(0, 35)]


for i in range(len(df_measured)):
    by_rms = df_measured.loc[i, 'By_RMS']
    bz_rms = df_measured.loc[i, 'Bz_RMS']
    calculated_angle = get_angle(filtered_database, bz_rms, by_rms)
    angles.append(calculated_angle)
    error.append(calculated_angle - degs[i])
    by_lst.append(by_rms)
    bz_lst.append(bz_rms)

# Create a DataFrame to store the results
df_error = pd.DataFrame({
    'angle': degs,
    'c_angle': angles,
    'error': error,
    'by': by_lst,
    'bz': bz_lst
})
df_error.to_csv('results11.csv', index=False)

# Load and filter the database for plotting
filtered_database = database[database['Angle'].between(0, 35)]

# Plotting
plt.figure(figsize=(14, 8))

by_lst_micro_tesla = [y * 1e6 for y in by_lst]
bz_lst_micro_tesla = [z * 1e6 for z in bz_lst]
database_y_micro_tesla = [y * 1e6 for y in filtered_database['y']]
database_z_micro_tesla = [z * 1e6 for z in filtered_database['z']]

# Trova i valori minimi e massimi tra tutti i dati scalati in microtesla
y_min = min(min(by_lst_micro_tesla), min(database_y_micro_tesla), min(bz_lst_micro_tesla), min(database_z_micro_tesla))
y_max = max(max(by_lst_micro_tesla), max(database_y_micro_tesla), max(bz_lst_micro_tesla), max(database_z_micro_tesla))

# Plot By data (misurati vs database) in microtesla
plt.subplot(2, 2, 1)
plt.plot(degs, by_lst_micro_tesla, marker='o', linestyle='-', color='b', label='By RMS (Misurato)')
plt.plot(filtered_database['Angle'], database_y_micro_tesla, marker='x', linestyle='--', color='g', label='By (Database)')
plt.xlabel('Gradi')
plt.ylabel('By (µT)')  # Etichetta asse y in microtesla
plt.title('By vs Gradi')
plt.ylim(99, 150)  # Imposta gli stessi limiti per l'asse y
plt.legend()

# Plot Bz data (misurati vs database) in microtesla
plt.subplot(2, 2, 2)
plt.plot(degs, bz_lst_micro_tesla, marker='o', linestyle='-', color='r', label='Bz RMS (Misurato)')
plt.plot(filtered_database['Angle'], database_z_micro_tesla, marker='x', linestyle='--', color='m', label='Bz (Database)')
plt.xlabel('Gradi')
plt.ylabel('Bz (µT)')  # Etichetta asse y in microtesla
plt.title('Bz vs Gradi')
plt.ylim(350, 456)  # Imposta gli stessi limiti per l'asse y
plt.legend()



# Add a new plot for desired vs calculated angles
plt.subplot(2, 1, 2)
plt.plot(degs, angles, marker='o', linestyle='-', color='b', label='Calculated Angle')
plt.xlabel('Desired Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.title('Desired vs Calculated Angles')
plt.grid(True)
plt.ylim([0, 35])  # Adjust y-limit for the correct angle range
plt.legend()

plt.plot([0, 35], [0, 35], color='r', linestyle='--', label='Ideal Line (0° to 35°)')


plt.tight_layout()
plt.show()
