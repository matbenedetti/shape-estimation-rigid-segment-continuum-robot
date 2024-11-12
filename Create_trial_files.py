import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from numpy import mod



# Function to calculate FFT magnitude and phase for a given data array
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



# Replace this with your sampling frequency
fs = 500  # Example sampling frequency

# Function to extract the 'x', 'y', 'z' columns and convert stringified lists into lists of floats
def extract_axis_data(df, axis):
    return df[axis].apply(eval).explode().astype(float).values

csv_files = ["04_Shape estimation/data_14102024/R0_AC1.csv", "04_Shape estimation/data_14102024/R1_AC1.csv", "04_Shape estimation/data_14102024/R2_AC1.csv", "04_Shape estimation/data_14102024/R3_AC1.csv", "04_Shape estimation/data_14102024/R4_AC1.csv", "04_Shape estimation/data_14102024/R5_AC1.csv"]  


# Prepare to hold the data for plotting
fft_values_all_files = []


# Process each CSV file
for file_idx, csv_file_path in enumerate(csv_files):
    df = pd.read_csv(csv_file_path)
    angles = df['Angle'].unique()
    fft_values_100Hz = []
    
    # Process data for each angle
    for idx, angle in enumerate(angles):
        # Filter the data for the current angle
        df_angle = df[df['Angle'] == angle]
        
        # Extract data for x, y, z axes
        Bxs = extract_axis_data(df_angle, 'x') * 1e6
        Bys = extract_axis_data(df_angle, 'y') * 1e6
        Bzs = extract_axis_data(df_angle, 'z') * 1e6
        
        
        # Calculate FFT for x, y, z axes
        freqs_x, Bxs_fft, Bxs_phase = calculate_fft(Bxs, fs)
        freqs_y, Bys_fft, Bys_phase = calculate_fft(Bys, fs)
        freqs_z, Bzs_fft, Bzs_phase = calculate_fft(Bzs, fs)


        Bxs_fft_100Hz = Bxs_fft[100]
        Bys_fft_100Hz = Bys_fft[100]
        Bzs_fft_100Hz = Bzs_fft[100]
        
        Bxs_phase_100Hz = Bxs_phase[100]
        Bys_phase_100Hz = Bys_phase[100]
        Bzs_phase_100Hz = Bzs_phase[100]

        phase_diff = Bxs_phase_100Hz - Bzs_phase_100Hz
        phase_diff = mod(phase_diff+ 3* np.pi / 2, 2*np.pi) -  (3* np.pi / 2)
        
        # Save the FFT values in a list as a dictionary for this angle
        fft_values_100Hz.append({
            'Angle': angle,
            'Bxs_fft_100Hz': Bxs_fft_100Hz,
            'Bys_fft_100Hz': Bys_fft_100Hz,
            'Bzs_fft_100Hz': Bzs_fft_100Hz,
            'Bxs_phase_100Hz': Bxs_phase_100Hz,
            'Bys_phase_100Hz': Bys_phase_100Hz,
            'Bzs_phase_100Hz': Bzs_phase_100Hz,
            'Phase Difference': phase_diff,
            'Trial': file_idx
        })
    df = pd.DataFrame(fft_values_100Hz)
    df.to_csv(f'04_Shape estimation/data_14102024/Tial{file_idx}_MP35mm.csv', index=False)