# plotting.py
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
df_error = pd.read_csv('results00.csv')

degs = df_error['angle']
angles = df_error['c_angle']
error = df_error['error']
by_lst = df_error['by']
bz_lst = df_error['bz']

# Plot results
plt.figure(figsize=(12, 8))

# Plot angles vs calculated angles
plt.subplot(2, 2, 1)
plt.plot(degs, angles, marker='o', linestyle='-', color='b', label='Calculated Angle')
plt.xlabel('Desired Angle (degrees)')
plt.ylabel('Calculated Angle (degrees)')
plt.title('Desired vs Calculated Angles')
plt.grid(True)
plt.ylim([-35, 35])
plt.legend()

# Plot angles vs errors
plt.subplot(2, 2, 2)
plt.plot(degs, error, marker='o', linestyle='-', color='r', label='Error')
plt.xlabel('Desired Angle (degrees)')
plt.ylabel('Error (degrees)')
plt.title('Error vs Desired Angle')
plt.grid(True)
plt.legend()

# Plot by vs angle
plt.subplot(2, 2, 3)
adjustment = df_error[df_error['angle'] == 0].iloc[0]
by_lst = (by_lst - adjustment['by']) * 1e6
plt.plot(degs, by_lst, marker='o', linestyle='-', color='g', label='By RMS')
plt.ylim(-1000, 1400)
plt.xlabel('Desired Angle (degrees)')
plt.ylabel('By RMS')
plt.title('By RMS vs Desired Angle')
plt.grid(True)
plt.legend()

# Plot bz vs angle
plt.subplot(2, 2, 4)
bz_lst = (bz_lst - adjustment['bz']) * 1e6
plt.plot(degs, bz_lst, marker='o', linestyle='-', color='m', label='Bz RMS')
plt.ylim(-1000, 1400)
plt.xlabel('Desired Angle (degrees)')
plt.ylabel('Bz RMS')
plt.title('Bz RMS vs Desired Angle')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('plot_results.png')  # Save the figure to a file
plt.show()
