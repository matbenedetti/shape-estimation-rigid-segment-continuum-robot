import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the 6 error files
folder_path = '/local/home/matildebenedetti/Downloads/Tesi_new/04_Shape estimation/data_14102024/'
error_files = ['Error_35mm.csv']

# Initialize a figure for plotting
plt.figure(figsize=(10, 6))

# Loop through each error file
for file in error_files:
    # Read the CSV file
    df = pd.read_csv(folder_path + file)

    # Assuming the columns are named 'angle' and 'angle_c' and we are interested in trial 0
    trial_data = df[df['Trial'] == 0]  # Filter for trial 0
    
    # Plot angle vs angle_c
    plt.plot(trial_data['angle'], trial_data['c_angle'], marker='o', label=file[:-4])  # Use file name without extension as label

# Customize the plot
plt.title('Angle vs Angle_c for Trial 0')
plt.xlabel('Angle (degrees)')
plt.ylabel('Angle_c (degrees)')
plt.legend()
plt.grid()
plt.axhline(0, color='black', lw=0.8, ls='--')  # Add a horizontal line at y=0
plt.axvline(0, color='black', lw=0.8, ls='--')  # Add a vertical line at x=0
plt.xlim(-40, 40)  # Adjust x-limits if necessary
plt.ylim(-40, 40)  # Adjust y-limits if necessary

# Show the plot
plt.show()

rms_errors = np.zeros((6, 9))
#read the file in a for loop
for idx_file, error_file in enumerate(error_files):
    error_data = pd.read_csv(folder_path + error_file)
    for trial in range(9):
        #get the data for the trial
        df_trial = error_data[error_data['Trial'] == trial]
        
        #get error
        error = df_trial['error']

        rms_error = np.sqrt(np.mean(error**2))
        if trial == 0:

            mean_error = np.mean(error)
            std_error = np.std(error)

            # Print the mean and standard deviation of the error for the trial
            print(f'File: {error_file}, Trial: {trial}, Mean Error: {mean_error:.4f}, Std Dev of Error: {std_error:.4f}')

        rms_errors[idx_file, trial] = rms_error
    mean_rms_error = np.mean(rms_errors[idx_file])
    #print(f'Mean RMS Error: {mean_rms_error}')

from scipy.stats import shapiro

# Example: Check normality of RMS errors for Setup 1
#check normality for all the setups
for idx in range(rms_errors.shape[0]):
    stat, p_value = shapiro(rms_errors[idx]) 
    #print(f'Setup {idx + 1} - p-value: {p_value}')

#i can assume normality for all the setups exept 5 

from scipy.stats import f_oneway

# Perform one-way ANOVA across all setups
stat, p_value = f_oneway(rms_errors[0], rms_errors[1], rms_errors[2],
                         rms_errors[3], rms_errors[5])

#print(f'ANOVA p-value: {p_value}')

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# Flatten the data for Tukey's HSD test
rms_errors_modified = np.delete(rms_errors, 4, axis=0)  # Remove row at index 3
errors = rms_errors_modified.flatten()
setups = np.repeat(np.arange(1, 6), 9)  # 6 setups, 9 trials each

tukey = pairwise_tukeyhsd(endog=errors, groups=setups, alpha=0.05)
#print(tukey)
