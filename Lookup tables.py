import numpy as np
import pandas as pd
import serial
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe

degs = np.arange(-36, 36, 1)
COM = "COM3"
BAUD = 4800
no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500

# Function to read data from the Arduino
def read_arduino(ser):
    is_finished = False
    while not is_finished:
        sinput = str(ser.readline().decode('utf-8')).rstrip() #PPP;x;y;z
        print(sinput)
        if len(sinput) > 0 and sinput == 'FM':
            is_finished = True
            return is_finished

# Initialize the serial connection
ard = serial.Serial(COM, BAUD, timeout=0.1)

# Cycle of 3 iterations
for cycle in range(1, 4):
    data_list = []
    for deg in degs:
        ard.write(bytes(str(deg), 'utf-8'))
        print(deg)
        motor_finished = read_arduino(ard)
        if motor_finished:
            databytes = w2bw_read_n_bytes((no_meas_per_frame) * no_bytes_per_meas, "COM12", fsample)

            print("Elaboration done")

            data = read_w2bw_tms320_data_syncframe(databytes)

            Bxs = np.array(data["Bxs"])
            Bys = np.array(data["Bys"])
            Bzs = np.array(data["Bzs"])

            bx_rms = np.sqrt(np.mean(Bxs**2))
            by_rms = np.sqrt(np.mean(Bys**2))
            bz_rms = np.sqrt(np.mean(Bzs**2))

            if deg != -36:
                data_list.append([bx_rms, by_rms, bz_rms, deg])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data_list, columns=['x', 'y', 'z', 'Angle'])
    df.to_csv(f'R{cycle}_1100_DC.csv', index_label='Index')

    print(f"Cycle {cycle} completed")
    print(df)

csv_files = ['R1_1100_DC.csv', 'R2_1100_DC.csv', 'R3_1100_DC.csv']
combined_data = pd.DataFrame()
for file in csv_files:
    df = pd.read_csv(file)
    combined_data = pd.concat([combined_data, df], axis=0)

mean_data = combined_data.groupby('Angle').mean().reset_index()

mean_data['Index'] = range(1, len(mean_data) + 1)

mean_data = mean_data[['Index', 'Angle', 'x', 'y', 'z']]

mean_data.to_csv('TOT_1100_DC.csv', index=False)

print(mean_data)