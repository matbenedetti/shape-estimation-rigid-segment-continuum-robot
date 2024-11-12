from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt
 
# Time array
f_sig = 10  # [Hz]
t = np.linspace(0, 1, f_sig*20, False)  # [s]


# Original signal
synthetic_signal = np.sin(2 * np.pi * 100 * time)
 
# Apply FFT
y_signal_fft = fft(y_signal)
 
# Get the corresponding frequency array
x_freq = np.fft.fftfreq(len(y_signal), t[1] - t[0])
 
# Apply inverse FFT without modifying the FFT result (to keep both magnitude and phase)
y_signal_inversed = ifft(y_signal_fft)
 
# sort the frequency array x_freq and the FFT result y_signal_fft
x_freq, y_signal_fft = zip(*sorted(zip(x_freq, y_signal_fft)))
 
 
# Plot the original and inversed signals in the time domain
plt.subplot(2, 1, 1)
plt.plot(t, y_signal, label='Original signal')
plt.plot(t, np.real(y_signal_inversed), label='Inversed signal', linestyle='--')
plt.legend()
plt.xlabel('Time [s]')
 
# Plot the magnitude of the FFT in the frequency domain
plt.subplot(2, 1, 2)
plt.plot(x_freq, np.abs(y_signal_fft), label='FFT Magnitude')
plt.legend()
plt.xlabel('Frequency [Hz]')
 
# Show the plots
plt.tight_layout()
plt.show()