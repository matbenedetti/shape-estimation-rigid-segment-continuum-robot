import smaract.scu as scu
from time import sleep
from w2bw_tms320 import w2bw_read_n_bytes,read_w2bw_tms320_data, read_w2bw_tms320_data_syncframe
from w2bw_tms320 import plot_frame_sig_and_fft

no_meas_per_frame = 16000
no_bytes_per_meas = 12
fsample = 500

wait_s = 0.05

wait_ms = round(wait_s * 1000)

# Function to wait for the motor to stop moving
def wait_for_finished_movement(deviceIndex, channelIndex):
    status = scu.GetStatus_S(deviceIndex, channelIndex)
    while status not in (scu.StatusCode.STOPPED, scu.StatusCode.HOLDING):
        status = scu.GetStatus_S(deviceIndex, channelIndex)
        sleep(wait_s)

def set_initial_position(deviceIndex, channels):
    print("Setting initial position as (0,0)")
    max_offsets_x_mm = 37
    max_offsets_y_mm = 100
    print("Init pos_x", scu.GetPosition_S(deviceIndex, 0) / 10000, "mm")
    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")
    
    pos_x = scu.GetPosition_S(deviceIndex, 0)
    strep = 1
    for i in range(max_offsets_x_mm * 10):
        scu.MovePositionRelative_S(deviceIndex, 0, -1000, 1000)
        wait_for_finished_movement(deviceIndex, 0)
        print("Index", i, "Channel", 0, ":", scu.GetPosition_S(deviceIndex, 0) / 10000, "mm")
        print("Difference:", (pos_x - scu.GetPosition_S(deviceIndex, 0)) / 10000, "mm")
        if (abs(pos_x - scu.GetPosition_S(deviceIndex, 0)) < 100):
            break
        else:
            pos_x = scu.GetPosition_S(deviceIndex, 0)

    scu.MovePositionRelative_S(deviceIndex, 1, max_offsets_y_mm * 1000 * 10, 1000)
    wait_for_finished_movement(deviceIndex, 1)
    print("Channel", 1, ":", scu.GetPosition_S(deviceIndex, 1) / 10000, "mm")
    
    

    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")

    set_to_zero_offsets = [0, 0 * 1000 * 10]
    for channelIndex, target_position in zip(channels, set_to_zero_offsets):
        scu.MovePositionRelative_S(deviceIndex, channelIndex, target_position, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)
        print("Channel", channelIndex, ":", scu.GetPosition_S(deviceIndex, channelIndex) / 10000, "mm")
    
    

# Move the motor to the target positions
def move_to_target_positions(deviceIndex, channels, target_positions):
    print("Moving to target positions...")
    """for channelIndex, target_position in zip(channels, target_positions):
        scu.MovePositionRelative_S(deviceIndex, channelIndex, target_position, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} moved to position {target_position}.")"""

    offset_x = target_positions[0]
    step_x = 1
    pos_x = scu.GetPosition_S(deviceIndex, 0)
    for i in range(round(offset_x / (1000 * 10)) + 1):  # Move from 0 mm to 10 mm in 1 mm step
        scu.MovePositionRelative_S(deviceIndex, 0, step_x * 1000 * 10, wait_ms)
        wait_for_finished_movement(deviceIndex, 0)
        print('Step:', i, "Channel:", 0, "Device index:", deviceIndex, "Step:", "+" if step_x > 0 else "", step_x, "mm")
        read_actual_position(deviceIndex, channels)
        if (abs(pos_x - scu.GetPosition_S(deviceIndex, 0)) < 200):
            break
        else:
            pos_x = scu.GetPosition_S(deviceIndex, 0)

    offset_y = target_positions[1]

    scu.MovePositionRelative_S(deviceIndex, 1, offset_y, wait_ms)
    wait_for_finished_movement(deviceIndex, 1)



def read_actual_position(deviceIndex, channels):
    for channelIndex in channels:
        position = scu.GetPosition_S(deviceIndex, channelIndex)
        print(f"Actual position of channel {channelIndex}: {position / 10:.1f} um")

# Main script

scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
deviceIndex = 0
channels = [0, 1]
target_positions = [0, -94004]  #Set your target positions here
#target_positions = [0, -10000]
#move_to_target_positions(deviceIndex, channels, target_positions)
set_initial_position(deviceIndex, channels)
move_to_target_positions(deviceIndex, channels, target_positions)

"""while True:
    user_input = input("Enter command: ").strip().lower()

    if user_input == 'a':
        read_actual_position(deviceIndex, channels)
    elif user_input == 'q':
        print("Exiting program.")
        break
    else:
        print("Invalid input. Please enter 'a' to read position or 'q' to quit.")"""
'''
channel = 0
mmStep = 1
pos_x = scu.GetPosition_S(deviceIndex, channel)
for i in range(37):  # Move from 0 mm to 10 mm in 1 mm step
    scu.MovePositionRelative_S(deviceIndex, channel, mmStep * 1000 * 10, wait_ms)
    wait_for_finished_movement(deviceIndex, channel)
    print('Step:', i, "Channel:", channel, "Device index:", deviceIndex, "Step:", "+" if mmStep > 0 else "", mmStep, "mm")
    read_actual_position(deviceIndex, channels)
    if (abs(pos_x - scu.GetPosition_S(deviceIndex, channel)) < 200):
        break
    else:
        pos_x = scu.GetPosition_S(deviceIndex, channel)
        '''

scu.ReleaseDevices()
print("SCU close.")
print("*******************************************************")
print("Done. Press return to exit.")
input()
