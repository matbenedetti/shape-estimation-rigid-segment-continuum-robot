import numpy as np
import pandas as pd
import serial
from scipy import optimize
import matplotlib.pyplot as plt
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe

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
angless = []
error_lst = []

df = pd.read_csv('AC_fft_0.csv', index_col = 0)

trial = df['Trial'].unique()
angles = df['Angle']

        
def get_angle(data, bn, bt): #capisci bn e bt
    lbnd = 0
    rbnd = 35
    angles = data['Angle']
    y_vals = data['Bys_fft_100Hz']
    z_vals = data['Bzs_fft_100Hz']
    wt = 0.1 #to optimize
    wn = 0.1
    def minFunction(angle):
        angn_cost = wn*abs(np.interp(angle, angles, z_vals)-bn)
        angt_cost = wt*abs(np.interp(angle, angles, y_vals)-bt)
        cost = angt_cost+angn_cost
        return cost
    min_angle = optimize.fminbound(minFunction, lbnd, rbnd)
    return min_angle



df_trial = df[df['Trial'] == 3]
for deg in angles:

    Bys = df_trial[df_trial['Angle'] == deg]['Bys_fft_100Hz'].values
    Bzs = df_trial[df_trial['Angle'] == deg]['Bzs_fft_100Hz'].values

    calculated_angle = get_angle(df_trial, Bys, Bzs)
    angless.append(calculated_angle)
    error.append(calculated_angle-deg)

        
#df_error = pd.DataFrame({'angle':degs, 'c_angle':angles, 'error': error, 'by': by_lst, 'bt':bz_lst})
#df_error.to_csv('results11.csv', index=False)
print(by_lst)

"""df_measured = pd.DataFrame({
    'Degrees': degs,
    'By_RMS': by_lst,
    'Bz_RMS': bz_lst
})
df_measured.to_csv('measured_data.csv', index=False)"""

# Load and filter the database
database = pd.read_csv('TOT_1100.csv', index_col=0)
filtered_database = database[database['Angle'].between(0, 35)]

# Plotting
plt.figure(figsize=(14, 8))

# Plot By data (measured vs. database)
plt.subplot(2, 2, 1)
plt.plot(degs, by_lst, marker='o', linestyle='-', color='b', label='By RMS (Measured)')
plt.plot(filtered_database['Angle'], filtered_database['y'], marker='x', linestyle='--', color='g', label='By (Database)')
plt.xlabel('Degrees')
plt.ylabel('By')
plt.title('By vs Degrees')
plt.legend()

# Plot Bz data (measured vs. database)
plt.subplot(2, 2, 2)
plt.plot(degs, bz_lst, marker='o', linestyle='-', color='r', label='Bz RMS (Measured)')
plt.plot(filtered_database['Angle'], filtered_database['z'], marker='x', linestyle='--', color='m', label='Bz (Database)')
plt.xlabel('Degrees')
plt.ylabel('Bz')
plt.title('Bz vs Degrees')
plt.legend()

plt.tight_layout()
plt.show()
