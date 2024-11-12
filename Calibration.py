from math import sqrt
import serial
import time
import smaract.scu as scu
from time import sleep
from w2bw_tms320 import w2bw_read_n_bytes,read_w2bw_tms320_data, read_w2bw_tms320_data_syncframe
from w2bw_tms320 import plot_frame_sig_and_fft

# Function to wait for the motor to stop moving
def wait_for_finished_movement(deviceIndex, channelIndex):
    status = scu.GetStatus_S(deviceIndex, channelIndex)
    while status not in (scu.StatusCode.STOPPED, scu.StatusCode.HOLDING):
        status = scu.GetStatus_S(deviceIndex, channelIndex)
        sleep(0.05)

# Initialize the motor position to zero
def move_to_initial_position(deviceIndex, channels):
    print("Setting initial position as (0,0)")
    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")

def move_to_target_positions(deviceIndex, channels, target_positions):
    print("Moving to position (3.6 mm left, 11.5 mm deep)")
    for channelIndex, target_position in zip(channels, target_positions):
        scu.MovePositionAbsolute_S(deviceIndex, channelIndex, target_position, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} moved to position {target_position}.")

def initialize_channel(deviceIndex, channelIndex):
    print(f"\nInitializing channel {channelIndex}...")
    print("SCU GetSensorPresent...")
    sensorPresent = scu.GetSensorPresent_S(deviceIndex, channelIndex)
    print("SCU GetSensorType...")
    sensorType = scu.GetSensorType_S(deviceIndex, channelIndex)
    
    if sensorPresent == scu.SENSOR_PRESENT:
        print(f"SCU Linear sensor present (configured type {sensorType}).")

def move_motor(deviceIndex, channelIndex, diff):
    scu.MovePositionRelative_S(deviceIndex, channelIndex, diff)
    wait_for_finished_movement(deviceIndex, channelIndex)
    position = scu.GetPosition_S(deviceIndex, channelIndex)
    print(f"Position on channel {channelIndex}: {position/10:.1f} um")

def read_actual_position(deviceIndex, channels):
    for channelIndex in channels:
        position = scu.GetPosition_S(deviceIndex, channelIndex)
        print(f"Actual position of channel {channelIndex}: {position / 10:.1f} um")



# Main script
scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
numOfDevices = scu.GetNumberOfDevices()
print("SCU number of devices:", numOfDevices)
no_meas_per_frame=1
no_bytes_per_meas=12
fsample=500
no_frames=1

deviceIndex = 0
channels = [0, 1]

for channelIndex in channels:
    initialize_channel(deviceIndex, channelIndex)

move_to_initial_position(deviceIndex, channels)

target_positions = [38015, -92052]
move_to_target_positions(deviceIndex, channels, target_positions)

read_actual_position(deviceIndex, channels)

max_sensor_value = -float('inf')
max_position = 0

for i in range(11):  # Move from 0 mm to 10 mm in 1 mm steps
    databytes=w2bw_read_n_bytes(no_frames*(no_meas_per_frame)*no_bytes_per_meas,"COM10",fsample)
    data=read_w2bw_tms320_data_syncframe(databytes)

    Bxs=data["Bxs"]
    Bys=data["Bys"]
    Bzs=data["Bzs"]
    tot = sqrt(Bys[0]**2 + Bxs[0]**2)

    if tot > max_sensor_value:
            max_sensor_value = tot
            max_position = i * 10000  # Convert mm to um

    if i < 10:  # Don't move the motor after the last measurement
        move_motor(deviceIndex, 0, 10000)  # Move motor by 1 mm (1000 um)

print(f"Maximum sensor X value: {max_sensor_value} at position: {max_position / 1000:.1f} mm")
move_motor(deviceIndex, 0, max_position - scu.GetPosition_S(deviceIndex, channelIndex))

scu.ReleaseDevices()
print("SCU close.")
print("*******************************************************")
print("Done. Press return to exit.")
input()