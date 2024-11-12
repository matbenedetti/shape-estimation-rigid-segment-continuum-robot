import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from numpy import polyfit

# Define your arctangent model function
def arctan_model(x, A, B, C, D):
    return A * np.arctan(B * x + C) + D

file_path = '/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/'
name_file = ['Test030.csv', 'Test035.csv',  'Test040.csv', 'Test045.csv', 'Test050.csv', 'Test055.csv']
labels = ['3 mm', '3.5 mm', '4 mm', '4.5 mm', '5 mm', '5.5 mm']  # Labels for plotting
folder_path = "/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_22102024/"
file_name = "Cubic Fitting Tan 0.csv"

full_path = folder_path + file_name

df = pd.read_csv(full_path)
a_listb = df['a']
b_listb = df['b']
c_listb = df['c']
d_listb = df['d']
# Read csv file 
plt.figure(figsize=(10, 8))
a_list = []
b_list = []
c_list = []
d_list = []
trial_list = []


# Iterate over each unique trial
for file_idx, file in enumerate(name_file):

    ab = a_listb[file_idx]
    bb = b_listb[file_idx]
    cb = c_listb[file_idx]
    db= d_listb[file_idx]

    fft_df = pd.read_csv(file_path + file)

    # Your data
    x_data = fft_df['Angle']  # The x-axis data (angles)
    y_data = fft_df['Phase Difference']  # The y-axis data (phase difference)

    # Fit the data to the arctangent model
    popt, _ = curve_fit(arctan_model, x_data, y_data, p0=[1, 1, 0, 0])  # Initial guesses for A, B, C, D

    # Extract fitted parameters
    A, B, C, D = popt
    print(f'Fitted parameters for Trial {file_idx}: A = {A}, B = {B}, C = {C}, D = {D}')

    # Generate fitted data for plotting
    x_fit = x_data
    y_fit = arctan_model(x_fit, *popt)
    straight_line = np.tan(y_data + 0.16)
    a, b, c, d = polyfit(x_data, straight_line, 3)
    fitted_line = a * x_data**3 + b * x_data**2 + c * x_data + d
    fitted_lineb = ab * x_data**3 + bb * x_data**2 + cb * x_data + db

    # save the parameters
    a_list.append(a)
    b_list.append(b)
    c_list.append(c)
    d_list.append(d)
    #trial_list.append(trail)
    #print the list tipe

    # Create a figure and subplots

    # Plot the original data and the fitted curve
    '''plt.subplot(3, 1, 1)
    plt.scatter(x_data, y_data,label=f'{labels[file_idx]} Raw Data', marker='o')  # Original data points
    plt.plot(x_fit, y_fit, label=f'{labels[file_idx]} Fitted', linestyle='--')  # Fitted arctan curve
    
    plt.subplot(3, 1, 2)
    fft_df['Bxs_Bzs_ratio'] = (y_fit) * (fft_df['Bzs_fft_100Hz'] / fft_df['Bzs_fft_100Hz'])
    plt.plot(x_fit, fft_df['Bxs_Bzs_ratio'], label=f' {labels[file_idx]} Adjusted Fitted') 

    # Plot Bxs/Bzs Ratio
    plt.subplot(3, 1, 3)'''
    plt.plot(x_data, straight_line, label=f'{labels[file_idx]} ')
    plt.plot(x_data, fitted_line, 'r--')
    #plt.plot(x_data, fitted_lineb, 'b-')



data = {
    'a': a_list,
    'b': b_list,
    'c': c_list,
    'd': d_list,
    #'trial': trial_list
}
df = pd.DataFrame(data)
print(df)
# Save the DataFrame to a CSV file
df.to_csv('/local/home/matildebenedetti/Downloads/OneDrive_2024-10-16/04_Shape estimation/shape-estimation-rigid-segment-continuum-robot/data_24102024/fitting.csv', index=False)

'''plt.subplot(3, 1, 3)
'''

# Add labels and a legend
plt.xlabel('Degrees')
plt.ylabel('Tan')
plt.title('Tan and its cubic fitting')
plt.legend()

# Show the plot
    

'''plt.subplot(3, 1, 1)


plt.axhline(y=np.pi/2, color='red', linestyle='--', label='π/2')
plt.axhline(y=-np.pi/2, color='red', linestyle='--', label='-π/2')
plt.title(f'Arctangent Fit')
plt.xlabel('Angle (degrees)')
plt.ylabel('Phase Difference (radians)')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)

plt.axhline(y=np.pi/2, color='red', linestyle='--', label='π/2')
plt.axhline(y=-np.pi/2, color='red', linestyle='--', label='-π/2')
plt.title(f'Arcatangent * Bxs/Bzs Ratio')
plt.xlabel('Angle (degrees)')
plt.ylabel('F(x) (radians)')
plt.grid(True)
plt.legend()'''

# Show the plot
plt.tight_layout()
plt.show()


