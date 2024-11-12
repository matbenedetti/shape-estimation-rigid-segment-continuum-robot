import numpy as np
import pandas as pd
import serial
import time
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe
import matplotlib.pyplot as plt


degs = np.arange(-36, 36, 1)
COM = "/dev/ttyACM0"
BAUD = 4800
no_meas_per_frame = 501
no_bytes_per_meas = 12
fsample = 500

# Function to read data from the Arduino
def read_arduino(ser):
    is_finished = False
    while not is_finished:
        sinput = str(ser.readline().decode('utf-8')).rstrip()
        print(sinput)
        if sinput == 'FM':
            is_finished = True
    return is_finished

# Initialize the serial connection
ard = serial.Serial(COM, BAUD, timeout=0.1)
time.sleep(2)

# Cycle of 6 iterations
for cycle in range(0, 6):
    data_list = []
    for deg in degs:
        ard.write(bytes(str(deg), 'utf-8'))
        print("desired", deg)
        motor_finished = read_arduino(ard)
        if motor_finished:
            databytes = w2bw_read_n_bytes((no_meas_per_frame) * no_bytes_per_meas, "/dev/ttyUSB0", fsample)

            print("Elaboration done")

            data = read_w2bw_tms320_data_syncframe(databytes)
            print("len data", len(data["Bxs"]))
            print("len data", len(data["Bys"]))
            print("len data", len(data["Bzs"]))

            #take only the first 500 measurements
            Bxs = (data["Bxs"])[:500]
            Bys = (data["Bys"])[:500]
            Bzs = (data["Bzs"])[:500]


            if deg != -36:
                data_list.append([Bxs, Bys, Bzs, deg])  

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_list, columns=['x', 'y', 'z', 'Angle'])
    df.to_csv(f'/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/R{cycle}_test1.csv', index_label='Index')

    print(f"Cycle {cycle} completed")
    print(df)

    # Wait for user input before proceeding to the next cycle
    input("Press Enter to start the next cycle...")

# Combine CSV files
csv_files = ['b1_1100.csv', 'b2_1100.csv', 'b3_1100.csv']
combined_data = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(file)
    combined_data = pd.concat([combined_data, df], axis=0)

# Calculate baseline
bx_0 = combined_data['x'].mean()
by_0 = combined_data['y'].mean()
bz_0 = combined_data['z'].mean()

# Calculate deviations from baseline
combined_data['Bx_deviation'] = combined_data['x'] - bx_0
combined_data['By_deviation'] = combined_data['y'] - by_0
combined_data['Bz_deviation'] = combined_data['z'] - bz_0

# Plot Bz - Bz,0 vs. Angle
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(combined_data['Angle'], combined_data['Bz_deviation'], 'o-', color='r')
plt.xlabel('Angle (degrees)')
plt.ylabel('Bz - Bz,0')
plt.title('Bz - Bz,0 vs. Angle')

# Plot By - By,0 vs. Angle
plt.subplot(1, 3, 2)
plt.plot(combined_data['Angle'], combined_data['By_deviation'], 'o-', color='g')
plt.xlabel('Angle (degrees)')
plt.ylabel('By - By,0')
plt.title('By - By,0 vs. Angle')

# Plot Bx - Bx,0 vs. Angle
plt.subplot(1, 3, 3)
plt.plot(combined_data['Angle'], combined_data['Bx_deviation'], 'o-', color='b')
plt.xlabel('Angle (degrees)')
plt.ylabel('Bx - Bx,0')
plt.title('Bx - Bx,0 vs. Angle')

plt.tight_layout()
plt.show()
