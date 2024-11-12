import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

csv_files = ["04_Shape estimation/data_01102024/R1_AC0.csv", "04_Shape estimation/data_01102024/R0_AC0.csv"]

# Function to extract the 'x', 'y', 'z' columns and convert stringified lists into lists of floats
def extract_axis_data(df, axis):
    return df[axis].apply(eval).explode().astype(float).values

# Replace this with your sampling frequency
fs = 500  # Example sampling frequency
time = np.arange(0, 499) / fs  # Generate time values for 500 samples

# Process each CSV file
for file_idx, csv_file_path in enumerate(csv_files):
    df = pd.read_csv(csv_file_path)
    angles = [-30, -10, -5, 0, 5, 10, 30]
    
    # Create a figure for plotting raw magnetic fields
    plt.figure(figsize=(15, 10))
    
    # Process data for each angle
    for idx, angle in enumerate(angles):
        # Filter the data for the current angle
        df_angle = df[df['Angle'] == angle]
        
        # Extract data for x, y, z axes
        Bxs = extract_axis_data(df_angle, 'x') * 1e6  # Convert to microteslas
        Bys = extract_axis_data(df_angle, 'y') * 1e6  # Convert to microteslas
        Bzs = extract_axis_data(df_angle, 'z') * 1e6  # Convert to microteslas
        
        # Plotting magnetic fields vs time for the current angle
        plt.subplot(3, 1, 1)  # Subplot for Bx
        plt.plot(np.arange(0, len(Bxs)) / fs, Bxs, label=f'Angle {angle}°')
        plt.title('Magnetic Field (Bx) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Bx (µT)')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

        plt.subplot(3, 1, 2)  # Subplot for By
        plt.plot(np.arange(0, len(Bys)) / fs, Bys, label=f'Angle {angle}°')
        plt.title('Magnetic Field (By) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('By (µT)')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

        plt.subplot(3, 1, 3)  # Subplot for Bz
        plt.plot(np.arange(0, len(Bzs)) / fs, Bzs, label=f'Angle {angle}°')
        plt.title('Magnetic Field (Bz) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Bz (µT)')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

    plt.tight_layout()
    plt.show()

    # Create a new figure for normalized magnetic fields
    plt.figure(figsize=(15, 10))
    
    # Normalize and plot the data for each angle
    for idx, angle in enumerate(angles):
        # Filter the data for the current angle
        df_angle = df[df['Angle'] == angle]
        
        # Extract data for x, y, z axes again to normalize each angle's data separately
        Bxs = extract_axis_data(df_angle, 'x') * 1e6  # Convert to microteslas
        Bys = extract_axis_data(df_angle, 'y') * 1e6  # Convert to microteslas
        Bzs = extract_axis_data(df_angle, 'z') * 1e6  # Convert to microteslas
        
        # Normalize the magnetic field data between 0 and 1
        Bxs_normalized = (Bxs - np.min(Bxs)) / (np.max(Bxs) - np.min(Bxs))
        Bys_normalized = (Bys - np.min(Bys)) / (np.max(Bys) - np.min(Bys))
        Bzs_normalized = (Bzs - np.min(Bzs)) / (np.max(Bzs) - np.min(Bzs))

        # Plotting normalized magnetic fields vs time for the current angle
        plt.subplot(3, 1, 1)  # Subplot for normalized Bx
        plt.plot(np.arange(0, len(Bxs_normalized)) / fs, Bxs_normalized, label=f'Angle {angle}°')
        plt.title('Normalized Magnetic Field (Bx) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Bx')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

        plt.subplot(3, 1, 2)  # Subplot for normalized By
        plt.plot(np.arange(0, len(Bys_normalized)) / fs, Bys_normalized, label=f'Angle {angle}°')
        plt.title('Normalized Magnetic Field (By) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized By')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

        plt.subplot(3, 1, 3)  # Subplot for normalized Bz
        plt.plot(np.arange(0, len(Bzs_normalized)) / fs, Bzs_normalized, label=f'Angle {angle}°')
        plt.title('Normalized Magnetic Field (Bz) vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Bz')
        plt.xlim(0, 0.2)  # Limit x-axis to 0.2 seconds
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid()

    plt.tight_layout()
    plt.show()
