import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
folder_path = "/local/home/matildebenedetti/Downloads/Tesi_new/04_Shape estimation/data_09102024/" 
file1 = folder_path + "heatinfluence1.csv"
file2 = folder_path + "heatinfluence2.csv"

fsample = 500

# Load CSV files into dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Check data to ensure consistency
logging.info(f"Loaded data from heatinfluence1.csv: {df1.shape[0]} iterations")
logging.info(f"Loaded data from heatinfluence2.csv: {df2.shape[0]} iterations")

# Extract FFT data
Bxs_100Hz_1, Bys_100Hz_1, Bzs_100Hz_1 = df1['Bxs_100Hz'], df1['Bys_100Hz'], df1['Bzs_100Hz']
Bxs_100Hz_2, Bys_100Hz_2, Bzs_100Hz_2 = df2['Bxs_100Hz'], df2['Bys_100Hz'], df2['Bzs_100Hz']

# Combine data for y-limits calculation
Bxs_fft_100Hz_list = np.concatenate([Bxs_100Hz_1, Bxs_100Hz_2])
Bys_fft_100Hz_list = np.concatenate([Bys_100Hz_1, Bys_100Hz_2])
Bzs_fft_100Hz_list = np.concatenate([Bzs_100Hz_1, Bzs_100Hz_2])

# Calculate dynamic y-limits
y_min = min(min(Bxs_fft_100Hz_list), min(Bys_fft_100Hz_list), min(Bzs_fft_100Hz_list))
y_max = max(max(Bxs_fft_100Hz_list), max(Bys_fft_100Hz_list), max(Bzs_fft_100Hz_list))
y_max = y_max + 0.1 * abs(y_max)
y_min = y_min - 0.1 * abs(y_min)

# Create time points based on the number of iterations
time_points_1 = np.arange(len(Bxs_100Hz_1)) * (250 / fsample)  # For heatinfluence1
time_points_2 = np.arange(len(Bxs_100Hz_2)) * (250 / fsample)  # For heatinfluence2

# Create subplots for comparison
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot Bxs comparison
axs[0].plot(time_points_1, Bxs_100Hz_1, label='Bxs 35°', marker='o', markersize=5, linestyle='--')
axs[0].plot(time_points_2, Bxs_100Hz_2, label='Bxs 0°', marker='o', markersize=5)
axs[0].set_title('Bxs FFT Magnitude @ 100Hz over Time')
axs[0].set_ylabel('FFT Magnitude')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(y_min, y_max)

# Plot Bys comparison
axs[1].plot(time_points_1, Bys_100Hz_1, label='Bys 35°', marker='o', markersize=5, linestyle='--')
axs[1].plot(time_points_2, Bys_100Hz_2, label='Bys 0°', marker='o', markersize=5)
axs[1].set_title('Bys FFT Magnitude @ 100Hz over Time')
axs[1].set_ylabel('FFT Magnitude')
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(y_min, y_max)

# Plot Bzs comparison
axs[2].plot(time_points_1, Bzs_100Hz_1, label='Bzs 35°', marker='o', markersize=5, linestyle='--')
axs[2].plot(time_points_2, Bzs_100Hz_2, label='Bzs 0°', marker='o', markersize=5)
axs[2].set_title('Bzs FFT Magnitude @ 100Hz over Time')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('FFT Magnitude')
axs[2].legend()
axs[2].grid(True)
axs[2].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
