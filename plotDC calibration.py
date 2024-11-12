import pandas as pd
import numpy as np
import smaract.scu as scu
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

scu.InitDevices(scu.SYNCHRONOUS_COMMUNICATION)

# Replace this with your sampling frequency
fs = 500  # Example sampling frequency

# Load data from the CSV file
csv_file_path = "CalibrationDC0y.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file_path)

Bxs = df['x'].values * 1e6
Bys = df['y'].values * 1e6
Bzs = df['z'].values * 1e6
motor_positions = df['motor_positions'].values


optimal_index = np.argmin(Bzs)

optimal_motor_position = motor_positions[optimal_index]
scu.MovePositionAbsolute_S(0, 0, optimal_motor_position, 1000)

# Print the results
print(f"Optimal index: {optimal_index}")
print(f"Minimized Bz value: {Bzs[optimal_index]}")
print(f"Maximized By value: {Bys[optimal_index]}")

scu.ReleaseDevices()

# Convert motor positions from device units to mm, if needed
motor_positions_mm = np.array(motor_positions) / 10000.0  # Assuming motor positions are in 0.1 micrometers
print(motor_positions_mm)
plt.figure(figsize=(12, 6))

# Plot Bz vs motor position
plt.subplot(1, 3, 1)
plt.scatter(motor_positions_mm[optimal_index], Bzs[optimal_index], color='red', label="Optimal Point", zorder=5)
plt.plot(motor_positions_mm,Bzs, label="Bz Data")
plt.xlabel("Position (mm)")
plt.ylabel("Bz (µT)")
plt.title("Bz Data (Minimized) vs Position")
plt.grid(True)
plt.legend()

# Plot By vs motor position
plt.subplot(1, 3, 2)
plt.scatter(motor_positions_mm[optimal_index], Bys[optimal_index], color='red', label="Optimal Point", zorder=5)
plt.plot(motor_positions_mm,Bys, label="By Data")
plt.xlabel("Position (mm)")
plt.ylabel("By (µT)")
plt.title("By Data (Maximized) vs Position")
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(motor_positions_mm[optimal_index], Bxs[optimal_index], color='red', label="Optimal Point", zorder=5)
plt.plot(motor_positions_mm,Bxs, label="Bx Data")
plt.xlabel("Position (mm)")
plt.ylabel("Bx (µT)")
plt.title("Bx Data")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
