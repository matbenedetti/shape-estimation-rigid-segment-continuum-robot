import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files
csv_files = ['R0_DC.csv', 'R1_DC.csv']

labels = ['2 mm', '2.5 mm', '3 mm', '3.5 mm', '4 mm', '4.5 mm']

# Initialize an empty DataFrame for combined data
combined_data = pd.DataFrame()

# Load each CSV file and add a 'Trial' column before concatenating
for idx, file in enumerate(csv_files):
    for idx, angle in enumerate(angles):
    # Filter the data for the current angle
    df_angle = df[df['Angle'] == angle]
    
    # Extract data for x, y, z axes
    Bxs = extract_axis_data(df_angle, 'x') * 1e6
    Bys = extract_axis_data(df_angle, 'y') * 1e6
    Bzs = extract_axis_data(df_angle, 'z') * 1e6

    #plot_frame_sig_and_fft([Bxs,Bys,Bzs], freqs = [92,98,104], fs = 500, N = 250)

    
    # Calculate FFT for x, y, z axes
    freqs_x, Bxs_fft = calculate_fft(Bxs, fs)
    freqs_y, Bys_fft = calculate_fft(Bys, fs)
    freqs_z, Bzs_fft = calculate_fft(Bzs, fs)
    df = pd.read_csv(file)
    df['Trial'] = idx  # Add a column indicating the trial number
    combined_data = pd.concat([combined_data, df], axis=0)

print(combined_data.iloc[35])

# Calculate baseline using the value at index 35 for each trial
combined_data['Bx_baseline'] = combined_data.groupby('Trial')['x'].transform(lambda g: g.iloc[35])
combined_data['By_baseline'] = combined_data.groupby('Trial')['y'].transform(lambda g: g.iloc[35])
combined_data['Bz_baseline'] = combined_data.groupby('Trial')['z'].transform(lambda g: g.iloc[35])

# Calculate deviations from baseline
combined_data['Bx_deviation'] = (combined_data['x'] - combined_data['Bx_baseline']) * 1e6
combined_data['By_deviation'] = (combined_data['y'] - combined_data['By_baseline']) * 1e6
combined_data['Bz_deviation'] = (combined_data['z'] - combined_data['Bz_baseline']) * 1e6

# Create subplots for Bx, By, and Bz
fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

# Plot Bx for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[0].plot(trial_data['Angle'], trial_data['Bx_deviation'], 'o-', label=labels[trial])

axs[0].set_ylabel('Bx (uT)')
axs[0].set_title('Bx vs. Angle for All Trials')
axs[0].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot By for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[1].plot(trial_data['Angle'], trial_data['By_deviation'], 'o-', label=labels[trial])

axs[1].set_ylabel('By (uT)')
axs[1].set_title('By vs. Angle for All Trials')
axs[1].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot Bz for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[2].plot(trial_data['Angle'], trial_data['Bz_deviation'], 'o-', label=labels[trial])

axs[2].set_xlabel('Angle (degrees)')
axs[2].set_ylabel('Bz (uT)')
axs[2].set_title('Bz vs. Angle for All Trials')
axs[2].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()