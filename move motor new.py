import smaract.scu as scu
from time import sleep

wait_s = 0.05
wait_ms = round(wait_s * 1000)

def wait_for_finished_movement(deviceIndex, channelIndex):
    status = scu.GetStatus_S(deviceIndex, channelIndex)
    while status not in (scu.StatusCode.STOPPED, scu.StatusCode.HOLDING):
        status = scu.GetStatus_S(deviceIndex, channelIndex)
        sleep(wait_s)


def set_initial_position(deviceIndex, channels):
 
    scu.MovePositionRelative_S(deviceIndex, 1, 1000 * 1000 * 10, 1000)
    wait_for_finished_movement(deviceIndex, 1)
    scu.MovePositionRelative_S(deviceIndex, 2, 1000 * 1000 * 10, 1000)
    wait_for_finished_movement(deviceIndex, 1)

    print("Setting initial position as (0,0)")

    print("Init pos_x", scu.GetPosition_S(deviceIndex, 0) / 10000, "mm")
    for channelIndex in channels:
        scu.SetZero_S(deviceIndex, channelIndex)
        print(f"Channel {channelIndex} set to zero position.")


def move_to_target_positions(deviceIndex, channels, target_positions):
    offset_y = target_positions[1]
    offset_x = target_positions[0]

    scu.MovePositionRelative_S(deviceIndex, 1, offset_y, wait_ms)
    wait_for_finished_movement(deviceIndex, 1)
    scu.MovePositionRelative_S(deviceIndex, 2, offset_x, wait_ms)
    wait_for_finished_movement(deviceIndex, 1)


def read_actual_position(deviceIndex, channels):
    for channelIndex in channels:
        position = scu.GetPosition_S(deviceIndex, channelIndex)
        print(f"Actual position of channel {channelIndex}: {position / 10:.1f} um")


scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
deviceIndex = 0
channels = [1, 2]
target_positions = [0, -94004]
set_initial_position(deviceIndex, channels)
#move_to_target_positions(deviceIndex, channels, target_positions)




scu.ReleaseDevices()
print("SCU close.")
print("*******************************************************")
print("Done. Press return to exit.")
input()