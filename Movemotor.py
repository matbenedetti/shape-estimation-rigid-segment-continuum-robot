#!/usr/bin/python3
import sys
import smaract.scu as scu
from time import sleep

def wait_for_finished_movement(deviceIndex, channelIndex):
    status = scu.GetStatus_S(deviceIndex, channelIndex)
    while status not in (scu.StatusCode.STOPPED, scu.StatusCode.HOLDING):
        status = scu.GetStatus_S(deviceIndex, channelIndex)
        sleep(0.05)

def read_actual_position(deviceIndex, channels):
    for channelIndex in channels:
        position = scu.GetPosition_S(deviceIndex, channelIndex)
        print(f"Actual position of channel {channelIndex}: {position / 10:.1f} um")

def initialize_channel(deviceIndex, channelIndex):
    print(f"\nInitializing channel {channelIndex}...")
    print("SCU GetSensorPresent...")
    sensorPresent = scu.GetSensorPresent_S(deviceIndex, channelIndex)
    print("SCU GetSensorType...")
    sensorType = scu.GetSensorType_S(deviceIndex, channelIndex)
    
    if sensorPresent == scu.SENSOR_PRESENT:
        print(f"SCU Linear sensor present (configured type {sensorType}).")
    
    if sensorPresent:
        print(f"{channelIndex}c Calibrate positioner")
        print(f"{channelIndex}r Reference positioner")
        print(f"{channelIndex}+ Move positioner up by 100 um")
        print(f"{channelIndex}- Move positioner down by 100 um")
    print(f"{channelIndex}q Quit channel {channelIndex}")

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

def process_command(deviceIndex, channels, key):
    try:
        channelIndex = int(key[0])
        command = key[1]
    except (ValueError, IndexError):
        print(f"Invalid command: {key}")
        return

    if channelIndex not in channels:
        print(f"Invalid channel: {channelIndex}")
        return

    sensorPresent = scu.GetSensorPresent_S(deviceIndex, channelIndex)

    if sensorPresent:
        if command == 'c':
            scu.CalibrateSensor_S(deviceIndex, channelIndex)
            wait_for_finished_movement(deviceIndex, channelIndex)
            print(f"Calibration done on channel {channelIndex}.")
        elif command == 'r':
            scu.MoveToReference_S(deviceIndex, channelIndex, 1000, False)
            wait_for_finished_movement(deviceIndex, channelIndex)
            print(f"Moved to reference on channel {channelIndex}.")
        elif command == 'a':
            read_actual_position(deviceIndex, channels)
        elif command == '+':
            scu.MovePositionRelative_S(deviceIndex, channelIndex, 1000, 1000)
            wait_for_finished_movement(deviceIndex, channelIndex)
            position = scu.GetPosition_S(deviceIndex, channelIndex)
            print(f"Position on channel {channelIndex}: {position/10:.1f} um")
        elif command == '-':
            scu.MovePositionRelative_S(deviceIndex, channelIndex, -1000, 1000)
            wait_for_finished_movement(deviceIndex, channelIndex)
            position = scu.GetPosition_S(deviceIndex, channelIndex)
            print(f"Position on channel {channelIndex}: {position/10:.1f} um")

    if command == 'q':
        print(f"Quitting operations for channel {channelIndex}.")
        return 'quit'

try:
    scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
    numOfDevices = scu.GetNumberOfDevices()
    print("SCU number of devices:", numOfDevices)

    deviceIndex = 0
    channels = [0, 1]

    for channelIndex in channels:
        initialize_channel(deviceIndex, channelIndex)

    move_to_initial_position(deviceIndex, channels)

    target_positions = [38015, -92052]
    move_to_target_positions(deviceIndex, channels, target_positions)

    read_actual_position(deviceIndex, channels)

    while True:
        key = input("Enter command (e.g., 0+, 1-, 0c, 1q): ")
        process_command(deviceIndex, channels, key)
        if process_command(deviceIndex, channels, key) == 'quit':
            break



finally:
    scu.ReleaseDevices()
    print("SCU close.")
    print("*******************************************************")
    print("Done. Press return to exit.")
    input()
