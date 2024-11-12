import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pandas as pd

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def plot_filtered_data(motor_positions_mm, Bx_filt, By_filt, Bz_filt, y_min, y_max):
    plt.figure(figsize=(12, 6))

    # Plot Bz vs motor position
    plt.subplot(1, 3, 1)
    plt.plot(Bz_filt, label="Bz Data")
    plt.xlabel("Position (mm)")
    plt.ylabel('FFT (uTrms)')
    plt.title('FFT z at 100Hz vs Angle')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)

    # Plot By vs motor position
    plt.subplot(1, 3, 2)
    plt.plot(By_filt, label="By Data")
    plt.xlabel("Position (mm)")
    plt.ylabel('FFT (uTrms)')
    plt.title('FFT y at 100Hz vs Angle')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)

    # Plot Bx vs motor position
    plt.subplot(1, 3, 3)
    plt.plot(Bx_filt, label="Bx Data")
    plt.xlabel("Position (mm)")
    plt.ylabel('FFT (uTrms)')
    plt.title('FFT x at 100Hz vs Angle')
    plt.grid(True)
    plt.legend()
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the data saved from main.py
    df = pd.read_csv('CalibrationAC0.csv')
    fft_df = pd.read_csv('CalibrationAC0_fft.csv')


    # Motor positions and FFT data
    motor_positions_mm = df['motor_positions'].values / 10000.0  # Convert to mm
    print (motor_positions_mm)
    Bxs_fft_100Hz =  fft_df['Bxs_fft_100Hz']
    Bys_fft_100Hz =  fft_df['Bys_fft_100Hz']
    Bzs_fft_100Hz =  fft_df['Bzs_fft_100Hz']


    fs = len(motor_positions_mm)/ (motor_positions_mm[-1] - motor_positions_mm[0])  
    Bx_filt = butter_lowpass_filter(fft_df['Bxs_fft_100Hz'], 200, fs)
    By_filt = butter_lowpass_filter(fft_df['Bys_fft_100Hz'], 200, fs)
    Bz_filt = butter_lowpass_filter(fft_df['Bzs_fft_100Hz'], 200, fs)
    print(fs)

    # Get y-axis min and max for consistent scaling
    y_min = min(fft_df[['Bxs_fft_100Hz', 'Bys_fft_100Hz', 'Bzs_fft_100Hz']].min())
    y_max = max(fft_df[['Bxs_fft_100Hz', 'Bys_fft_100Hz', 'Bzs_fft_100Hz']].max())

    # Plot the filtered data
    plot_filtered_data(motor_positions_mm, Bxs_fft_100Hz, Bys_fft_100Hz, Bzs_fft_100Hz, y_min, y_max)
