import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the previously saved CSV
df = pd.read_csv("C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/data_06112024/TOT2_45.csv")

# List to store non-filtered FFT values from all files
fft_values_non_filtered = []

# Loop over the unique trial values to collect non-filtered FFT values
for file_idx in df['Trial'].unique():
    # Collect non-filtered values for each trial
    trial_data = df[df['Trial'] == file_idx]
    fft_values_non_filtered.append(trial_data[['Angle', 'Bxs_fft_100Hz', 'Bys_fft_100Hz', 'Bzs_fft_100Hz', 'Phase Difference']])

# Combine all FFT values into a single DataFrame for each trial (non-filtered)
df_combined = pd.concat(fft_values_non_filtered, keys=range(len(df['Trial'].unique())))

# Group by 'Angle' and take the mean across all trials (files)
df_mean = df_combined.groupby('Angle').mean().reset_index()

# Save the mean values to a CSV file
output_path = "C:/Users/Benedetti/OneDrive - Scuola Superiore Sant'Anna/shape-estimation-rigid-segment-continuum-robot (2)/LUT/LUT2_45.csv"
df_mean.to_csv(output_path, index=False)

# Optional: Print the first few rows to verify the result
print(df_mean.head())


# Plotting both individual and mean FFT magnitudes
plt.figure(figsize=(10, 12))

# Plot for Bxs_fft_100Hz
plt.subplot(3, 1, 1)
for file_idx in df['Trial'].unique():
    trial_data = df[df['Trial'] == file_idx]
    plt.plot(trial_data['Angle'], trial_data['Bxs_fft_100Hz'], label=f'Trial {file_idx} Bxs FFT 100Hz', marker='o', linestyle='--')
# Add mean plot
plt.plot(df_mean['Angle'], df_mean['Bxs_fft_100Hz'], label='Mean Bxs FFT 100Hz', marker='o', color='black', linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Bxs FFT Magnitude')
plt.title('Bxs FFT Magnitude at 100Hz (Mean and Individual Trials)')
plt.grid(True)
plt.legend()

# Plot for Bys_fft_100Hz
plt.subplot(3, 1, 2)
for file_idx in df['Trial'].unique():
    trial_data = df[df['Trial'] == file_idx]
    plt.plot(trial_data['Angle'], trial_data['Bys_fft_100Hz'], label=f'Trial {file_idx} Bys FFT 100Hz', marker='o', linestyle='--')
# Add mean plot
plt.plot(df_mean['Angle'], df_mean['Bys_fft_100Hz'], label='Mean Bys FFT 100Hz', marker='o', color='black', linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Bys FFT Magnitude')
plt.title('Bys FFT Magnitude at 100Hz (Mean and Individual Trials)')
plt.grid(True)
plt.legend()

# Plot for Bzs_fft_100Hz
plt.subplot(3, 1, 3)
for file_idx in df['Trial'].unique():
    trial_data = df[df['Trial'] == file_idx]
    plt.plot(trial_data['Angle'], trial_data['Bzs_fft_100Hz'], label=f'Trial {file_idx} Bzs FFT 100Hz', marker='o', linestyle='--')
# Add mean plot
plt.plot(df_mean['Angle'], df_mean['Bzs_fft_100Hz'], label='Mean Bzs FFT 100Hz', marker='o', color='black', linewidth=2)
plt.xlabel('Angle (degrees)')
plt.ylabel('Bzs FFT Magnitude')
plt.title('Bzs FFT Magnitude at 100Hz (Mean and Individual Trials)')
plt.grid(True)
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
