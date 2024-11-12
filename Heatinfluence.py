import smaract.scu as scu
import pandas as pd
from time import sleep
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe
import numpy as np
import logging
from scipy.fftpack import fft
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
name_file = "heatinfluence1.csv"
no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500


Bxs, Bys, Bzs = np.array([]), np.array([]), np.array([])
data_iterations = 210
folder_path = "/local/home/matildebenedetti/Downloads/Tesi_new/04_Shape estimation/data_09102024/" 
full_path = folder_path + name_file


def calculate_fft(data, fs):
    T = len(data) / fs
    df = 1 / T  # Frequency resolution
    
    data = np.array(data)
    
    data_fft = fft(data)[:len(data)//2]  # Take half due to symmetry
    
    # Magnitude and phase
    magnitude = np.abs(data_fft)
    phase = np.angle(data_fft)
    
    # Frequency array
    freqs = np.fft.fftfreq(len(data), d=1/fs)[:len(data)//2]
    
    return freqs, magnitude, phase

# Function to collect data without moving the motor
def collect_data_without_movement(data_iterations, no_meas_per_frame):
    Bxs_fft_100Hz_list, Bys_fft_100Hz_list, Bzs_fft_100Hz_list = [], [], []
    for i in range(data_iterations):
        databytes = w2bw_read_n_bytes(501 * no_bytes_per_meas, "/dev/ttyUSB0", fsample)
        data = read_w2bw_tms320_data_syncframe(databytes)
        Bxs = np.array(data["Bxs"][:500]) * 1e6
        Bys = np.array(data["Bys"][:500]) * 1e6
        Bzs = np.array(data["Bzs"][:500]) * 1e6

        print(len(Bxs), len(Bys), len(Bzs))

        freqs_x, Bxs_fft, Bxs_phase = calculate_fft(Bxs, fsample)
        freqs_y, Bys_fft, Bys_phase = calculate_fft(Bys, fsample)
        freqs_z, Bzs_fft, Bzs_phase = calculate_fft(Bzs, fsample)

        idx_100Hz = np.argmin(np.abs(freqs_x - 100))  # Get index closest to 100Hz

        Bxs_fft_100Hz = Bxs_fft[idx_100Hz]
        Bys_fft_100Hz = Bys_fft[idx_100Hz]
        Bzs_fft_100Hz = Bzs_fft[idx_100Hz]

        Bxs_fft_100Hz_list.append(Bxs_fft_100Hz)
        Bys_fft_100Hz_list.append(Bys_fft_100Hz)
        Bzs_fft_100Hz_list.append(Bzs_fft_100Hz)
        
        logging.info(f"Data round {i+1}/{data_iterations} collected.")
    
    return Bxs_fft_100Hz_list, Bys_fft_100Hz_list, Bzs_fft_100Hz_list

# Collect data 190 times, no motor movement
Bxs_fft_100Hz_list, Bys_fft_100Hz_list, Bzs_fft_100Hz_list = collect_data_without_movement(data_iterations, no_meas_per_frame)

# Calculate time points for each iteration
time_points = np.arange(data_iterations) * (250 / fsample)  # Each iteration takes 250/500 seconds

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Calculate global y-limits
y_min = min(min(Bxs_fft_100Hz_list), min(Bys_fft_100Hz_list), min(Bzs_fft_100Hz_list))
y_max = max(max(Bxs_fft_100Hz_list), max(Bys_fft_100Hz_list), max(Bzs_fft_100Hz_list))
y_max = y_max + 0.1 * abs(y_max)

# Plot Bxs FFT
axs[0].plot(time_points, Bxs_fft_100Hz_list, label='Bxs FFT @ 100Hz', marker='o', markersize=5, color='blue')
axs[0].set_title('Bxs FFT Magnitude @ 100Hz over Time')
axs[0].set_ylabel('FFT Magnitude')
axs[0].set_ylim(y_min, y_max)  # Set y-axis limits
axs[0].legend()
axs[0].grid(True)

# Plot Bys FFT
axs[1].plot(time_points, Bys_fft_100Hz_list, label='Bys FFT @ 100Hz', marker='o', markersize=5, color='green')
axs[1].set_title('Bys FFT Magnitude @ 100Hz over Time')
axs[1].set_ylabel('FFT Magnitude')
axs[1].set_ylim(y_min, y_max)  # Set y-axis limits
axs[1].legend()
axs[1].grid(True)

# Plot Bzs FFT
axs[2].plot(time_points, Bzs_fft_100Hz_list, label='Bzs FFT @ 100Hz', marker='o', markersize=5, color='red')
axs[2].set_title('Bzs FFT Magnitude @ 100Hz over Time')
axs[2].set_xlabel('Time (seconds)')
axs[2].set_ylabel('FFT Magnitude')
axs[2].set_ylim(y_min, y_max)  # Set y-axis limits
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Save collected data to a CSV file
df = pd.DataFrame({'Bxs_100Hz': Bxs_fft_100Hz_list, 'Bys_100Hz': Bys_fft_100Hz_list, 'Bzs_100Hz': Bzs_fft_100Hz_list})
df.to_csv(full_path, index_label='Iteration')
logging.info(f"Data saved to {full_path}")


