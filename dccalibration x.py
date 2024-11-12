import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smaract.scu as scu
from motor_functions import (
    calibrate_sensor, move_to_stop_mechanical, set_initial_position,
    move_channel_1, move_channel_0
)

# Main script
scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)
deviceIndex = 0
channels = [0]

# Calibration and movement
calibrate_sensor(deviceIndex, 0)
calibrate_sensor(deviceIndex, 1)
calibrate_sensor(deviceIndex, 2)

move_to_stop_mechanical(0, 0)
#move_to_stop_mechanical(0, 2)

set_initial_position(deviceIndex, channels)
#move_channel_1(deviceIndex, 1, -3000)
move_channel_1(deviceIndex, 0, 27139)

# Collect and process data
Bxs, Bys, Bzs, cycle, motor_positions = move_channel_0(deviceIndex, 0, 0.1)

# Save data to CSV
df = pd.DataFrame({'x': Bxs, 'y': Bys, 'z': Bzs, 'motor_positions': motor_positions})
df.to_csv('CalibrationDC0y.csv', index_label='Index')

# Convert and analyze data
Bxs = df['x'].values * 1e6
Bys = df['y'].values * 1e6
Bzs = df['z'].values * 1e6
motor_positions = df['motor_positions'].values

optimal_index = np.argmin(Bzs)
optimal_motor_position = motor_positions[optimal_index]
scu.MovePositionAbsolute_S(0, 2, optimal_motor_position, 1000)

# Print results
print(f"Optimal index: {optimal_index}")
print(f"Maximized Bz value: {Bzs[optimal_index]}")
print(f"Minimized By value: {Bys[optimal_index]}")

scu.ReleaseDevices()

# Convert motor positions from device units to mm, if needed
motor_positions_mm = np.array(motor_positions) / 10000.0  # Assuming motor positions are in 0.1 micrometers
print(motor_positions_mm)

# Plot results
plt.figure(figsize=(12, 6))

# Plot Bz vs motor position
plt.subplot(1, 3, 1)
plt.plot(Bzs, label="Bz Data")
plt.scatter(motor_positions_mm[optimal_index], Bzs[optimal_index], color='red', label="Optimal Point")
plt.xlabel("Position (mm)")
plt.ylabel("Bz (µT)")
plt.title("Bz Data (Minimized) vs Position")
plt.grid(True)
plt.legend()

# Plot By vs motor position
plt.subplot(1, 3, 2)
plt.plot(Bys, label="By Data")
plt.scatter(motor_positions_mm[optimal_index], Bys[optimal_index], color='red', label="Optimal Point")
plt.xlabel("Position (mm)")
plt.ylabel("By (µT)")
plt.title("By Data (Maximized) vs Position")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(Bxs, label="Bx Data")
plt.scatter(motor_positions_mm[optimal_index], Bxs[optimal_index], color='red', label="Optimal Point")
plt.xlabel("Position (mm)")
plt.ylabel("Bx (µT)")
plt.title("Bx Data")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
