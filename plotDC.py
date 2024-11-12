import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load data from the CSV file
csv_files = ["04_Shape estimation/data_07102024/R0_DC0.csv", "04_Shape estimation/data_07102024/R1_DC0.csv", "04_Shape estimation/data_07102024/R2_DC0.csv"]  # Replace with your actual file path
labels = ['2.5 mm', '3 mm', '3.5 mm', '3.5 mm', '4 mm', '4.5 mm']

# Initialize an empty DataFrame for combined data
combined_data = pd.DataFrame()

# Extract the 'x', 'y', 'z' columns and convert stringified lists into lists of floats
def extract_axis_data(df, axis):
    return df[axis].apply(eval).explode().astype(float).values



# Process data for each angle
for trial_idx, file in enumerate(csv_files):
    df = pd.read_csv(file)
    angles = df['Angle']
    df_mean = []

    for idx, angle in enumerate(angles):
        # Filter the data for the current angle
        df_angle = df[df['Angle'] == angle]
        
        # Extract data for x, y, z axes
        Bxs = extract_axis_data(df_angle, 'x') * 1e6
        Bys = extract_axis_data(df_angle, 'y') * 1e6
        Bzs = extract_axis_data(df_angle, 'z') * 1e6

        Bxs_mean = np.mean(Bxs)
        Bys_mean = np.mean(Bys)
        Bzs_mean = np.mean(Bzs)

        
        # Save the FFT values in a list as a dictionary for this angle
        df_mean.append({
            'Angle': angle,
            'Bxs_fft_100Hz': Bxs_mean,
            'Bys_fft_100Hz': Bys_mean,
            'Bzs_fft_100Hz': Bzs_mean,
            'Trial': trial_idx
        })
    df_mean = pd.DataFrame(df_mean)
    print(df_mean)
    combined_data = pd.concat([combined_data, df_mean], axis=0)

# save the combined data to a CSV file
combined_data.to_csv('04_Shape estimation/data_07102024/Trial_DC0_35mm.csv', index=False)




'''print(combined_data.iloc[35])

# Calculate baseline using the value at index 35 for each trial
combined_data['Bx_baseline'] = combined_data.groupby('Trial')['Bxs_mean'].transform(lambda g: g.iloc[35])
combined_data['By_baseline'] = combined_data.groupby('Trial')['Bys_mean'].transform(lambda g: g.iloc[35])
combined_data['Bz_baseline'] = combined_data.groupby('Trial')['Bzs_mean'].transform(lambda g: g.iloc[35])

# Calculate deviations from baseline
combined_data['Bx_deviation'] = (combined_data['Bxs_mean'] - combined_data['Bx_baseline'])
combined_data['By_deviation'] = (combined_data['Bys_mean'] - combined_data['By_baseline'])
combined_data['Bz_deviation'] = (combined_data['Bzs_mean'] - combined_data['Bz_baseline'])

# Create subplots for Bx, By, and Bz
fig, axs = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

y_min = min(combined_data['Bx_deviation'].min(), combined_data['By_deviation'].min(), combined_data['Bz_deviation'].min())
y_max = max(combined_data['Bx_deviation'].max(), combined_data['By_deviation'].max(), combined_data['Bz_deviation'].max())

common_ylim = (y_min, y_max)

# Plot Bx for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[0].plot(trial_data['Angle'], trial_data['Bx_deviation'], 'o-', label=labels[trial])

axs[0].set_ylabel('Bx (uT)')
axs[0].set_title('Bx vs. Angle for All Trials')
axs[0].set_ylim(common_ylim) 
axs[0].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot By for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[1].plot(trial_data['Angle'], trial_data['By_deviation'], 'o-', label=labels[trial])

axs[1].set_ylabel('By (uT)')
axs[1].set_title('By vs. Angle for All Trials')
axs[1].set_ylim(common_ylim) 
axs[1].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot Bz for all trials
for trial in combined_data['Trial'].unique():
    trial_data = combined_data[combined_data['Trial'] == trial]
    axs[2].plot(trial_data['Angle'], trial_data['Bz_deviation'], 'o-', label=labels[trial])

axs[2].set_xlabel('Angle (degrees)')
axs[2].set_ylabel('Bz (uT)')
axs[2].set_title('Bz vs. Angle for All Trials')
axs[2].set_ylim(common_ylim) 
axs[2].legend(title='Trial', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()'''