from w2bw_tms320 import w2bw_read_n_bytes, read_w2bw_tms320_data_syncframe
from w2bw_tms320 import plot_frame_sig_and_fft
import numpy as np


no_meas_per_frame = 250
no_bytes_per_meas = 12
fsample = 500

databytes = w2bw_read_n_bytes((no_meas_per_frame)*no_bytes_per_meas,"/dev/ttyUSB0",fsample)

print(databytes)

print("Elaboration done")

data = read_w2bw_tms320_data_syncframe(databytes)

time_step = 1/2000

time = np.linspace(0, (no_meas_per_frame - 1) * time_step, no_meas_per_frame)  # Time array from 0 to the total duration

# Generate synthetic sinusoidal signal at 100 Hz
synthetic_signal = np.sin(2 * np.pi * 100 * time)

Bxs = data["Bxs"]
Bys = data["Bys"]
Bzs  =data["Bzs"]


print(Bxs)
print(Bys)
print(Bzs)


print("using %d samples"%(len(Bxs)))
plot_frame_sig_and_fft([Bxs,Bys,Bzs], freqs = [92,98,104], fs = 500, N = no_meas_per_frame)


