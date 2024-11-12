# rotate 2.5 mm from max of manual stage
#Channel 2 successfully moved to position 10.0988 mm.
#Channel 1 successfully moved to position -10.297 mm.
import smaract.scu as scu
import pandas as pd
from time import sleep
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


name_file = "Calibration00y.csv"
no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500
wait_s = 0.05
wait_ms = round(wait_s * 1000)

Bxs, Bys, Bzs = np.array([]), np.array([]), np.array([])
motor_positions = []
fft_values_100Hz = []
folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_22102024"
full_path = folder_path + name_file




def calibrate_sensor(device_index, channel_index, timeout=30):
    try:
        # Start calibration routine
        result = scu.CalibrateSensor_S(device_index, channel_index)
        logging.info(f"Calibration started for device {device_index}, channel {channel_index}. Result: {result}")
        
        # Set a timeout for calibration to avoid an infinite loop
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = scu.GetStatus_S(device_index, channel_index)
            if status == scu.StatusCode.STOPPED:
                logging.info(f"Calibration complete for device {device_index}, channel {channel_index}.")
                return True
            time.sleep(0.1)  # Small delay to avoid busy-waiting
        
        logging.warning(f"Calibration timed out after {timeout} seconds for device {device_index}, channel {channel_index}.")
        return False
    except Exception as e:
        logging.error(f"Error during calibration: {e}")
        return False

def wait_for_finished_movement(deviceIndex, channelIndex):
    status = scu.GetStatus_S(deviceIndex, channelIndex)
    while status not in (scu.StatusCode.STOPPED, scu.StatusCode.HOLDING):
        status = scu.GetStatus_S(deviceIndex, channelIndex)
        sleep(wait_s)

def move_to_stop_mechanical(deviceIndex, channelIndex):

    while scu.GetStatus_S(deviceIndex, channelIndex) != scu.StatusCode.STOPPED:
        scu.MovePositionRelative_S(deviceIndex, channelIndex, -5000, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)

    print(f"Channel {channelIndex} reached mechanical stop.")

def set_initial_position(deviceIndex, channels):
    print("Setting initial position as (0,0)")
    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")



def move_motor(deviceIndex, channelIndex, position):
    step_size = int(round(position * 1000 * 10))
    pos_ = scu.GetPosition_S(deviceIndex, channelIndex)
    scu.MovePositionRelative_S(deviceIndex, channelIndex, step_size, 1000)
    wait_for_finished_movement(deviceIndex, channelIndex)
    pos_pos = scu.GetPosition_S(deviceIndex, channelIndex)
    print(f"from {pos_} moved to position {pos_pos} mm.")

def calculate_fft(data, fs):
    T = len(data) / fs
    df = 1 / T  # Frequency resolution
    
    data = np.array(data)    
    data_fft = fft(data)[:len(data)//2]  # Take half due to symmetry
    data_fft = np.abs(data_fft)
    
    # Frequency array
    freqs = np.fft.fftfreq(len(data), d=1/fs)[:len(data)//2]
    
    return freqs, data_fft

def fft_dataframe(df, fsample):
    for pos in motor_positions:
        df_pos = df[df['motor_positions'] == pos]
        #extract data for x, y, z axes
        Bxs = df_pos['x'].values
        Bys = df_pos['y'].values
        Bzs = df_pos['z'].values
        freqs_x, Bxs_fft = calculate_fft(Bxs, fsample)
        freqs_y, Bys_fft = calculate_fft(Bys, fsample)
        freqs_z, Bzs_fft = calculate_fft(Bzs, fsample)


        fft_values_100Hz.append({
            'Bzs_fft' : Bzs_fft,
            'Bys_fft' : Bys_fft,
            'Bxs_fft' : Bxs_fft,
            'motor_positions': motor_positions,
            'Bxs_fft_100Hz': Bxs_fft[100],
            'Bys_fft_100Hz': Bys_fft[100],
            'Bzs_fft_100Hz': Bzs_fft[100]
        })

    return pd.DataFrame(fft_values_100Hz)


def mean_dataframe(df):
    df_mean = []
    motor_positions = []
    motor_positions = df['motor_positions'].unique()
    for idx, pos in enumerate(motor_positions):
        df_pos = df[df['motor_positions'] == pos]
        #extract data for x, y, z axes
        Bxs = df_pos['x'].values
        Bys = df_pos['y'].values
        Bzs = df_pos['z'].values
        Bxs_mean = np.mean(Bxs)
        Bys_mean = np.mean(Bys)
        Bzs_mean = np.mean(Bzs)
        df_mean.append({
            'Bzs_mean' : Bzs_mean,
            'Bys_mean' : Bys_mean,
            'Bxs_mean' : Bxs_mean,
            'motor_positions': pos
        })

    return pd.DataFrame(df_mean)


def collect_data_and_move(deviceIndex, channelIndex, step_size_mm, name_file):
    Bxs, Bys, Bzs = np.array([]), np.array([]), np.array([])
    pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
    counter = 0
    while True:
        if 'x' in name_file:
            move_motor(deviceIndex, channelIndex, -step_size_mm)
        else: 
            move_motor(deviceIndex, channelIndex, step_size_mm)
        databytes = w2bw_read_n_bytes((no_meas_per_frame) * no_bytes_per_meas, "/dev/ttyUSB0", fsample)
        data = read_w2bw_tms320_data_syncframe(databytes)
        Bxs = np.append(Bxs, np.array(data["Bxs"]) * 1e6)
        Bys = np.append(Bys, np.array(data["Bys"]) * 1e6)
        Bzs = np.append(Bzs, np.array(data["Bzs"]) * 1e6)

        pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
        motor_positions.extend([pos_x] * len(data["Bxs"]))
        print(f"Moved channel {channelIndex} by {step_size_mm} mm. Current position: {pos_x:.2f} mm")

        #plot_frame_sig_and_fft([Bxs,Bys,Bzs], freqs = [92,98,104], fs = 500, N = 250)
        counter += 1

        if scu.GetStatus_S(deviceIndex, channelIndex) == scu.StatusCode.STOPPED or counter > 185:
            print(f"Channel {channelIndex} reached mechanical stop.")
            break
    return Bxs, Bys, Bzs, motor_positions

# Main script

scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
deviceIndex = 0
channels = [1, 2]

#to comment after fist calibration
'''calibrate_sensor(deviceIndex, 1)
scu.SetZero_S(deviceIndex, 1)
move_motor(deviceIndex, 1, -10.297)

calibrate_sensor(deviceIndex, 2)
scu.SetZero_S(deviceIndex, 2)
move_motor(deviceIndex, 2, 10.0988)'''



#move_motor(deviceIndex, 1, 2*0.1)
#move_motor(deviceIndex, 2, 2*0.1)




#for channelIndex in channels:
#    move_to_stop_mechanical(deviceIndex, channelIndex)

if  'x' in name_file:
    calibrate_sensor(deviceIndex, 1)
    #calibrate_sensor(deviceIndex, 2)
    #set_initial_position(deviceIndex, channels)


    #move_motor(deviceIndex, 2, 7)
    Bxs, Bys, Bzs, motor_positions = collect_data_and_move(deviceIndex, 1, 0.1, name_file)


if 'y' in name_file:
    calibrate_sensor(deviceIndex, 2)
    Bxs, Bys, Bzs, motor_positions = collect_data_and_move(deviceIndex, 2, 0.1, name_file)

df = pd.DataFrame({'x': Bxs, 'y': Bys, 'z': Bzs, 'motor_positions': motor_positions})
df.to_csv(full_path, index_label='Index')
#read csv name_file
df = pd.read_csv(full_path)


df_mean = mean_dataframe(df)
print (df)
df_mean.to_csv(folder_path + 'CalibrationDC0x_mean', index_label='Index')



  
if  'x' in name_file:
    Bzs = df_mean['Bzs_mean']
    Bys = df_mean['Bys_mean']
    Bxs = df_mean['Bxs_mean']

    # Find two local maxima in Bz
    peaks_Bz, _ = find_peaks(Bzs, distance=5)
    # Get the peak heights
    peak_heights = Bzs[peaks_Bz]

    # Get the indices of the two highest peaks
    sorted_peak_indices = np.argsort(peak_heights)[-2:]
    
    sorted_peak_indices = np.array(sorted_peak_indices)
    print(sorted_peak_indices)

    # Extract the locations of the two highest peaks
    first_Bz_peak = peaks_Bz[sorted_peak_indices[-1]] 
    second_Bz_peak = peaks_Bz[sorted_peak_indices[-2]]
    # Extract the corresponding peak locations

    print(f"First Bz peak: {first_Bz_peak}, Second Bz peak: {second_Bz_peak}")


    # Find two local minima in By (invert By to find minima as peaks)
    inverted_By = -Bys
    peaks_By, _ = find_peaks(inverted_By, distance=5)
    first_By_peak = peaks_By[0]
    second_By_peak = peaks_By[1]
    print(f"First By peak: {first_By_peak}, Second By peak: {second_By_peak}")
    # take data between the two peaks
    cutted_Bzs = Bzs[first_Bz_peak : second_Bz_peak]
    print(Bzs)
    cutted_Bys = Bys[first_By_peak:second_By_peak]

    '''# Calculate mean and standard deviation for bz and by
    bz_mean = np.mean(cutted_Bzs)
    bz_std = np.std(cutted_Bzs)

    by_mean = np.mean(cutted_Bys)
    by_std = np.std(cutted_Bys)

    # Apply Z-score normalization
    bz_norm = (cutted_Bzs - bz_mean) / bz_std
    by_norm = (cutted_Bys - by_mean) / by_std

    # Compute the cost function using the normalized values
    cost =  0.5 * np.abs(cutted_Bzs) - np.abs(cutted_Bys)'''

    # Find the index of the minimum cost
    min_cost_index = np.argmin(cutted_Bzs)
    min_cost_index = min_cost_index +  first_Bz_peak


if 'y' in name_file:
    Bzs = df_mean['Bzs_mean']
    Bys = df_mean['Bys_mean']
    Bxs = df_mean['Bxs_mean']

    # Find two local maxima in Bz
    peaks_Bz, _ = find_peaks(Bzs, distance=5)
    first_Bz_peak = peaks_Bz[0]
    second_Bz_peak = peaks_Bz[1]
    print(f"First Bz peak: {first_Bz_peak}, Second Bz peak: {second_Bz_peak}")

    cutted_Bzs = Bzs[first_Bz_peak:second_Bz_peak]


    # Find the index of the minimum cost
    min_cost_index = np.argmin(cutted_Bzs)
    min_cost_index = min_cost_index + first_Bz_peak









# Convert motor positions from device units to mm, if needed
univ_motor_posiion = np.array(df_mean['motor_positions']) / 10000.0
print('N. steps:', len(univ_motor_posiion))
optimal_motor_position = univ_motor_posiion[min_cost_index]
if 'x' in name_file:
    channelIndex = 1
else:
    channelIndex = 2
current_position = scu.GetPosition_S(deviceIndex, channelIndex)    
print(f"Moving channel {channelIndex} to target: {optimal_motor_position} mm")
scu.MovePositionAbsolute_S(deviceIndex, channelIndex, int (optimal_motor_position*10000), 1000)
wait_for_finished_movement(deviceIndex, channelIndex)

# Check the position after movement
current_position = scu.GetPosition_S(deviceIndex, channelIndex)
print(f"Current position of channel {channelIndex}: {current_position} um")
        
print(f"Channel {channelIndex} successfully moved to position {optimal_motor_position} mm.")




scu.ReleaseDevices()



# DC Case: Plot mean values for each axis (x, y, z)
plt.figure(figsize=(12, 6))

# Get the min and max values for y-axis across Bx, By, and Bz means
y_min = min(df_mean[['Bxs_mean', 'Bys_mean', 'Bzs_mean']].min())
y_max = max(df_mean[['Bxs_mean', 'Bys_mean', 'Bzs_mean']].max())

# Plot mean z vs Position
plt.subplot(1, 3, 1)
plt.plot(univ_motor_posiion, df_mean['Bzs_mean'], label="Bz")
plt.scatter([univ_motor_posiion[first_Bz_peak], univ_motor_posiion[second_Bz_peak]],
        [df_mean['Bzs_mean'].iloc[first_Bz_peak], df_mean['Bzs_mean'].iloc[second_Bz_peak]],
        color='red', label="Bz Peaks", zorder=5)
plt.scatter([univ_motor_posiion[min_cost_index]],
        [df_mean['Bzs_mean'].iloc[min_cost_index]],
        color='blue', label="Best pos", zorder=5)
plt.xlabel("Position (mm)")
plt.ylabel('Bz (uT)')
plt.title(' Bz vs Position')
plt.grid(True)
plt.legend()
plt.ylim(y_min, y_max)

# Plot mean y vs Position
plt.subplot(1, 3, 2)
plt.plot(univ_motor_posiion, df_mean['Bys_mean'], label="By")

if 'x' in name_file:
    plt.scatter([univ_motor_posiion[first_By_peak], univ_motor_posiion[second_By_peak]],
            [df_mean['Bys_mean'].iloc[first_By_peak], df_mean['Bys_mean'].iloc[second_By_peak]],
            color='red', label="By Peaks", zorder=5)
plt.scatter([univ_motor_posiion[min_cost_index]],
        [df_mean['Bys_mean'].iloc[min_cost_index]],
        color='blue', label="Best pos", zorder=5) 
plt.xlabel("Position (mm)")
plt.ylabel('By (uT)')
plt.title(' By vs Position')
plt.grid(True)
plt.legend()
plt.ylim(y_min, y_max)

# Plot mean x vs Position
plt.subplot(1, 3, 3)
plt.plot(univ_motor_posiion, df_mean['Bxs_mean'], label="Bx")
plt.scatter([univ_motor_posiion[min_cost_index]],
    [df_mean['Bxs_mean'].iloc[min_cost_index]],
    color='blue', label="Best pos", zorder=5)
plt.xlabel("Position (mm)")
plt.ylabel('Bx (uT)')
plt.title(' Bx vs Position')
plt.grid(True)
plt.legend()
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()

