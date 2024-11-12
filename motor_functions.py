import smaract.scu as scu
import pandas as pd
from time import sleep
from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe
import numpy as np
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500
wait_s = 0.05
wait_ms = round(wait_s * 1000)

Bxs = []
Bys = []
Bzs = []
motor_positions = []

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
    pos = 5000
    if channelIndex == 2:
        pos = -5000

    # Start moving the device
    scu.MovePositionRelative_S(deviceIndex, channelIndex, pos, 1000)
    wait_for_finished_movement(deviceIndex, channelIndex)

    stop_count = 0  # Track how many times the device has stopped

    while stop_count < 2:
        # Check if the device has stopped
        if scu.GetStatus_S(deviceIndex, channelIndex) == scu.StatusCode.STOPPED:
            stop_count += 1
            print(f"Device stopped {stop_count} time(s)")

            if stop_count == 2:
                print("Device stopped twice, stopping further attempts.")
                break

        # Try moving the device again if not yet stopped twice
        scu.MovePositionRelative_S(deviceIndex, channelIndex, pos, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)

    print(f"Channel {channelIndex} reached mechanical stop.")

def set_initial_position(deviceIndex, channels):
    print("Setting initial position as (0,0)")
    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")

def move_channel_1(deviceIndex, channelIndex, position, tolerance=10, max_attempts=4):
    target_position = position * 1000
    scu.MovePositionRelative_S(deviceIndex, channelIndex, target_position, 1000)
    wait_for_finished_movement(deviceIndex, channelIndex)
    
    pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
    attempts = 0
    
    while abs(pos_x - target_position) > tolerance and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}: Adjusting position...")
        scu.MovePositionRelative_S(deviceIndex, channelIndex, target_position - pos_x, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)
        pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
    
    if abs(pos_x - target_position) <= tolerance:
        print(f"Channel {channelIndex} moved to position {position} mm.")
    else:
        print(f"Failed to move Channel {channelIndex} to position {position} mm after {attempts} attempts.")

def move_channel_0(deviceIndex, channelIndex, step_size_mm):
    pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
    cycle = 1
    step_size = int(round(step_size_mm * 1000 * 10))
    while True:
        scu.MovePositionRelative_S(deviceIndex, channelIndex, step_size, 1000)
        wait_for_finished_movement(deviceIndex, channelIndex)
        databytes = w2bw_read_n_bytes((no_meas_per_frame) * no_bytes_per_meas, "/dev/ttyUSB0", fsample)
        data = read_w2bw_tms320_data_syncframe(databytes)
        Bxs.extend(data["Bxs"])
        Bys.extend(data["Bys"])
        Bzs.extend(data["Bzs"])
        pos_x = scu.GetPosition_S(deviceIndex, channelIndex)
        motor_positions.extend([pos_x] * len(data["Bxs"]))
        print(f"Moved channel {channelIndex} by {step_size_mm} mm. Current position: {pos_x:.2f} mm")
        cycle += 1
        if scu.GetStatus_S(deviceIndex, channelIndex) == scu.StatusCode.STOPPED:
            print(f"Channel {channelIndex} reached mechanical stop.")
            break
    return Bxs, Bys, Bzs, cycle, motor_positions
