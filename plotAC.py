import pandas as pd
import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from numpy import mod



# Function to calculate FFT magnitude and phase for a given data array
def calculate_fft(data, fs):
    T = len(data) / fs
    df = 1 / T  # Frequency resolution
    
    data = np.array(data)
    
    data_fft = fft(data)[:len(data)//2]  # Take half due to symmetry
    
    # Magnitude and phase
    magnitude = np.abs(data_fft)/len(data)
    magnitude[1:] = 2 * magnitude[1:] 
    phase = np.angle(data_fft)
    
    # Frequency array
    freqs = np.fft.fftfreq(len(data), d=1/fs)[:len(data)//2]
    
    return freqs, magnitude, phase



# Replace this with your sampling frequency
fs = 500  # Example sampling frequency

# Function to extract the 'x', 'y', 'z' columns and convert stringified lists into lists of floats
def extract_axis_data(df, axis):
    return df[axis].apply(eval).explode().astype(float).values

# List of CSV files to process
csv_files = ["C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_06112024/Raw data/R2_data35.csv"]

colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Colors for plotting
labels = ['(0,0)', '(0.1,-0.1)', '(0.1,0.1)', '(-0.1,0.1)', '(-0.1,-0.1)']  # Labels for plotting


# Prepare to hold the data for plotting
fft_values_all_files = []

# Moving average window size
window_size = 1


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

        #pad the data to have the same length of 500
        if len(Bxs) < 500:
            Bxs = np.pad(Bxs, (0, 500 - len(Bxs)))
        if len(Bys) < 500:
            Bys = np.pad(Bys, (0, 500 - len(Bys)))
        if len(Bzs) < 500:
            Bzs = np.pad(Bzs, (0, 500 - len(Bzs)))

        
        
        # Calculate FFT for x, y, z axes
        freqs_x, Bxs_fft, Bxs_phase = calculate_fft(Bxs, fs)
        freqs_y, Bys_fft, Bys_phase = calculate_fft(Bys, fs)
        freqs_z, Bzs_fft, Bzs_phase = calculate_fft(Bzs, fs)

        #time vector
        time = np.linspace(0, len(Bxs)/fs, len(Bxs))


        Bxs_fft_100Hz = Bxs_fft[100]
        Bys_fft_100Hz = Bys_fft[100]
        Bzs_fft_100Hz = Bzs_fft[100]

        if angle in [-26, -22, -16, -15, -14, -12, 19, 21]:
            print(f'FFT of Bz at Angle {angle}')
            print(Bzs_fft_100Hz)
        
        Bxs_phase_100Hz = Bxs_phase[100]
        Bys_phase_100Hz = Bys_phase[100]
        Bzs_phase_100Hz = Bzs_phase[100]

        # Plot for the y-axis FFT magnitude



        phase_diff = Bxs_phase_100Hz - Bzs_phase_100Hz

        offset_phase = np.pi/2*3
        phase_diff = mod(phase_diff + offset_phase , 2*np.pi) -  offset_phase + np.pi/2
        
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
            
    
    # Add the data for this file to the list
    fft_values_all_files.append(pd.DataFrame(fft_values_100Hz))

# Concatenate all DataFrames in the list
df = pd.concat(fft_values_all_files, ignore_index=True)


# Save the updated dataframe to a new CSV file
df.to_csv(f"C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_06112024/Testcoarse_30.csv", index_label='Index')


plt.figure(figsize=(17, 12))

# Determine the global min and max of FFT values across all axes
min_value = min([df['Bxs_fft_100Hz'].min(), df['Bys_fft_100Hz'].min(), df['Bzs_fft_100Hz'].min()])
max_value = max([df['Bxs_fft_100Hz'].max(), df['Bys_fft_100Hz'].max(), df['Bzs_fft_100Hz'].max()])

# Optional: Set a buffer to avoid points being too close to the plot edges
buffer = 0.01 * (max_value - min_value)
y_min = min_value - 4*buffer
y_max = max_value + buffer

# Loop over files and plot magnitude data for each one
for file_idx, fft_df in enumerate(fft_values_all_files):
    color = colors[file_idx % len(colors)]  # Cycle through colors
    
    # Plot for the x-axis FFT magnitude
    plt.subplot(2, 2, 1)
    plt.plot(fft_df['Angle'], fft_df['Bxs_fft_100Hz'], label=f'{labels[file_idx]} ', marker='o', color=color)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT (uT)')
    plt.title('Bxs FFT Magnitude at 100Hz', pad=15)  # Add padding to the title
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()

    # Plot for the y-axis FFT magnitude
    plt.subplot(2, 2, 2)
    plt.plot(fft_df['Angle'], fft_df['Bys_fft_100Hz'], label=f'{labels[file_idx]}', marker='o', color=color)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT (uT)')
    plt.title('Bys FFT Magnitude at 100Hz', pad=15)  # Add padding to the title
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()

    # Plot for the z-axis FFT magnitude
    plt.subplot(2, 2, 3)
    plt.plot(fft_df['Angle'], fft_df['Bzs_fft_100Hz'] , label=f'{labels[file_idx]}', marker='o', color=color)       
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT (uT)')
    plt.title('Bzs FFT Magnitude at 100Hz', pad=15)  # Add padding to the title
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
 

# Adjust layout and add some padding
plt.tight_layout(pad=3.0)  # Add extra padding between subplots

# Optionally, adjust further using subplots_adjust for fine control
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust horizontal and vertical space between plots

#plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Create a new figure for the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

fft_df = fft_values_all_files[0]
color = colors[file_idx % len(colors)]  # Cycle through colors
    
# Get the data for each component and angle
angles = fft_df['Angle']
Bx = fft_df['Bxs_fft_100Hz']
By = fft_df['Bys_fft_100Hz']
Bz = fft_df['Bzs_fft_100Hz']

    

    # Plot the vectors in 3D
ax.plot(angles, Bz, Bx, By, label=f'{labels[file_idx]} from calibration')

# Labeling the axes with better clarity
ax.set_xlabel('Angle (degrees)')
ax.set_ylabel('Bz FFT Magnitude (uT)')
ax.set_zlabel('Bx FFT Magnitude (uT)')
ax.set_title('3D Plot of Bx, By, Bz FFT Magnitudes Over Angles')

# Adding a legend
ax.legend()

plt.show()





'''for file_idx, fft_df in enumerate(fft_values_all_files):
    Bxs_phase = fft_df['Bxs_phase_100Hz']  # Assuming Bxs_phase_100Hz is a column name
    Bzs_phase = fft_df['Bzs_phase_100Hz']  # Assuming Bzs_phase_100Hz is a column name
    
    for i in range(int(len(Bxs_phase)/2)):
        #plt.figure(figsize=(14, 10))
        
        # Correct the subtraction operation, ensuring array dimensions match
        B_diff = Bxs_phase[:len(Bxs_phase) - i] - Bzs_phase[i:]
        print(Bxs_phase[:len(Bxs_phase) - i], angles[:len(Bxs_phase) - i])

        offset_phase = np.pi/2*3
        phase_diff = mod(B_diff + offset_phase , 2*np.pi) -  offset_phase + np.pi/2
        #eliminate Nan values
        phase_diff = phase_diff[~np.isnan(phase_diff)]
        # reset index
        phase_diff = phase_diff.reset_index(drop=True)
        print(phase_diff)
        
        fft_norm = fft_df['Phase Difference'] 


        # Plot the difference between Bxs and Bzs
        plt.plot(fft_df['Angle'], fft_norm, label=f'{labels[file_idx]} from calibration', marker='o')
        plt.plot(angles[:len(phase_diff)], phase_diff, label='Bxs FFT (uT)', color='r')
        
        # Ensure the angle is properly defined
        angle = angles[i]
        plt.title(f'Bx at Angle {angle}°')
        plt.xlabel('Angle (°)')
        plt.ylabel('Phase (radians)')
        
        plt.legend()
        plt.grid(True)

        plt.axhline(y=-np.pi/2, color='red', linestyle='--', label='y = -π/2')  # Dashed line at -π
        plt.axhline(y=np.pi/2, color='blue', linestyle='--', label='y = π/2')
        
        # Show the plot
        plt.show()'''




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

# Helper function to format y-axis labels as multiples of pi
def format_func(value, tick_number):
    N = int(np.round(value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi$"
    elif N == -1:
        return r"-$\pi$"
    else:
        return r"${0}\pi$".format(N)

# First plot: Phase Difference
fig, axes = plt.subplots(2, 1, figsize=(17, 12))  # 2 rows, 1 column for phase difference

# Loop over files and plot phase data for each one in a separate subplot
for file_idx, fft_df in enumerate(fft_values_all_files):
    ax = axes[0]
    ax_tan = axes[1]
    color = colors[file_idx % len(colors)]  # Cycle through colors
    
    # Normalize phase difference to 0-2pi
    fft_norm = fft_df['Phase Difference'] 
    #fft_norm = fft_df['Bxs_phase_100Hz'] 
    
    # Plot phase difference in each subplot
    ax.plot(fft_df['Angle'], fft_norm, label=f'{labels[file_idx]} from calibration', marker='o', color=color)
    '''ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))  # Set major ticks at multiples of pi
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))  # Format ticks as multiples of pi'''
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Phase Difference (radians)')
    ax.set_title(f'FFT Phase Difference vs Angle for {labels[file_idx]}')
    ax.grid(True)

    ax_tan.plot(fft_df['Angle'], np.tan(fft_norm), label=f'Tan of {labels[file_idx]}', marker='o', color=color)
    
    # Add labels and title for each subplot
    ax_tan.set_xlabel('Angle (degrees)')
    ax_tan.set_ylabel('Tan of Phase Difference')
    ax_tan.set_title(f'Tangent of FFT Phase Difference vs Angle for {labels[file_idx]}')

    # Display legend
    ax_tan.legend()

ax.axhline(y=-np.pi/2, color='red', linestyle='--', label='y = -π/2')  # Dashed line at -π
ax.axhline(y=np.pi/2, color='blue', linestyle='--', label='y = π/2')

# Display legend
ax.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the phase difference plot
#plt.show()



for file_idx, fft_df in enumerate(fft_values_all_files):
    color = colors[file_idx % len(colors)]  # Cycle through colors
    
    # Plot for the x-axis FFT magnitude
    plt.subplot(2, 2, 1)
    Bzs_fft_diff = fft_df['Bzs_fft_100Hz'].diff()  # Calculate the difference between consecutive elements
    plt.plot(fft_df['Angle'][1:], Bzs_fft_diff[1:], label=f'{labels[file_idx]} FFT diff', marker='o', color=color)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT Magnitude Difference')
    plt.title('Bzs FFT Magnitude Difference')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    Bys_fft_diff = fft_df['Bys_fft_100Hz'].diff()  # Calculate the difference between consecutive elements
    plt.plot(fft_df['Angle'][1:], Bys_fft_diff[1:], label=f'{labels[file_idx]} FFT diff', marker='o', color=color)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT Magnitude Difference')
    plt.title('Bys FFT Magnitude Difference')
    plt.grid(True)
    plt.legend()

    # Plot for the z-axis FFT magnitude
    plt.subplot(2, 2, 3)
    Bxs_fft_diff = fft_df['Bxs_fft_100Hz'].diff()  # Calculate the difference between consecutive elements
    plt.plot(fft_df['Angle'][1:], Bxs_fft_diff[1:], label=f'{labels[file_idx]} FFT diff', marker='o', color=color)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('FFT Magnitude Difference')
    plt.title('Bxs FFT Magnitude Difference')
    plt.grid(True)
    plt.legend()


plt.tight_layout()
#plt.show()




'''

while True:
    angle = input('Insert the angle of the magnet (or type "CC" to exit): ')
    
    # Check if user wants to exit the loop
    if angle == 'CC':
        break
    
    # Convert the input to a numeric value (assuming angle input is a number)
    try:
        angle = float(angle)
    except ValueError:
        print("Invalid input. Please enter a valid number for the angle.")
        continue
    
    number_file = input('Insert number file: ')
    # Find the file index for the given angle
    file_path = csv_files[int(number_file)]
    df = pd.read_csv(file_path)

    # Extract data for the given angle
    Bxs = extract_axis_data(df, 'x') * 1e6
    Bys = extract_axis_data(df, 'y') * 1e6
    Bzs = extract_axis_data(df, 'z') * 1e6

    # Calculate FFT
    freqs_x, Bxs_fft, Bxs_phase = calculate_fft(Bxs, fs)
    freqs_y, Bys_fft, Bys_phase = calculate_fft(Bys, fs)
    freqs_z, Bzs_fft, Bzs_phase = calculate_fft(Bzs, fs)

    print (f'FFT of Bx at Angle {angle}')
    print(Bxs_fft[100])
    print (f'FFT of By at Angle {angle}')
    print(Bys_fft[100])
    print (f'FFT of Bz at Angle {angle}')
    print(Bzs_fft[100])

    # Plot FFT result for Bx
    plt.figure(figsize=(17, 12))
    plt.subplot(3, 1, 1)
    plt.plot(freqs_x[1:], Bxs_fft[1:], label='Bxs FFT Magnitude', color='r')
    plt.title(f'FFT of Bx at Angle {angle}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Plot FFT result for By
    plt.subplot(3, 1, 2)
    plt.plot(freqs_y[1:], Bys_fft[1:], label='Bys FFT Magnitude', color='g')
    plt.title(f'FFT of By at Angle {angle}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Plot FFT result for Bz
    plt.subplot(3, 1, 3)
    plt.plot(freqs_z[1:], Bzs_fft[1:], label='Bzs FFT Magnitude', color='b')
    plt.title(f'FFT of Bz at Angle {angle}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()'''

