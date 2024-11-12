import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import serial
import time
import csv
from scipy import optimize
from animationPlot import AnimationPlot

FILENAME_TEMP = "X_?_0.2A_200_Ymm" #R2_male_0.2A_200_13.0mm
# DISTANCES = ['10.5', '11.5', '12.5', '13.0']
# SENSING = ['4.5', '3.5', '2.5', '2']
DISTANCES = ['11.0', '11.5', '11.70', '11.85', '12.0'] #Height element of Test Setup

#DISTANCES = ['10.5', '11.0', '11.5', '12.0', '12.5', '13.0'] #Height element of Test Setup
SENSING = ['4.5', '4.0', '3.5', '3.0', '2.5', '2'] #Equivalent sensing radius to DISTANCES

KEYS = ['R1', 'R2', 'R3'] #Key of Measurement Cycle
PARTS = ['male', 'female'] #Available Parts
# COLORS = ["#2E8B57", "#FF6347", "#4682B4", "#DAA520"]

COLORS = ["#2E8B57", "#FF6347", "#4682B4", "#DAA520", "#6A5ACD", "#FFD700", "#C71585"]

COM = "COM3"
BAUD = 2000000

#Get File name
# Out: file name as string
def get_filename(file_temp, distance, key, part):
    file = file_temp
    file = file.replace('?', part)
    file = file.replace('Y', distance)
    file = file.replace('X', key)
    return file

#Get data
# Out: List of all keys as data frame
def get_data(file_template, keys, distance, part):
    data_lst = []
    for key in keys:
        file = get_filename(file_template, distance, key, part)
        data_lst.append(pd.read_csv(file, index_col=0))
    return data_lst

#Find max of each row
# In: dataframe
# Out: max value of each position as Dataframe
def get_max_df(df):
    rows = list(set(df.index.tolist()))
    max = pd.DataFrame()
    for row in rows:
        m = df.loc[row].max().to_frame().T
        max = pd.concat([max, m], ignore_index=True)
    return max

#Find min of each row
# In: dataframe
# Out: min value of each position as Dataframe
def get_min_df(df):
    rows = list(set(df.index.tolist()))
    min = pd.DataFrame()
    for row in rows:
        m = df.loc[row].min().to_frame().T
        min = pd.concat([min, m], ignore_index=True)
    return min

#Find average of each row
# In: dataframe
# Out: average value of each position as Dataframe
def get_average_df(df):
    rows = list(set(df.index.tolist()))
    avg = pd.DataFrame()
    for row in rows:
        m = df.loc[row].mean().to_frame().T
        avg = pd.concat([avg, m], ignore_index=True)
    return avg

#Subtract all values by inital value
# In: dataframe
# Out: Subtracted values as dataframe
def get_variation_to_zero(df):
    return df.subtract(df.iloc[0,:])

#Plot on single dataframe
def plot_single(df):
    mean_df = get_average_df(df)
    max_df = abs(mean_df-get_max_df(df))
    min_df = abs(mean_df-get_min_df(df))
    mean_df = get_variation_to_zero(mean_df)
    plt.scatter(mean_df['angle'],mean_df['z'], label='B_n')
    plt.errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']], fmt='o')
    plt.scatter(mean_df['angle'],mean_df['y'], label='B_t', marker='s')
    plt.errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']], fmt='s')
    plt.legend()
    plt.grid()
    plt.show()

# Get data
# Out: List of all keys and distances as data frame
def get_multiple_data(file_temp, keys, distances, part):
    data_lst = []
    for distance in distances:
        all_data = get_data(FILENAME_TEMP, KEYS, distance, part)
        data = pd.concat(all_data)
        data_lst.append(data)
    return data_lst

#Plot data for all distances
def plot_multiple(data_set):
    fig, ax = plt.subplots(2)
    fig.suptitle('B-Field Variation to Inital Value of Test Setup V4: 0.2A, 200 Windings, male')
    fig.supylabel('Magnetix Flux Density [uT]')
    patches = []

    i = 0
    for data in data_set:
        patch = mpatches.Patch(color=COLORS[i], label=SENSING[i])
        patches.append(patch)
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
        mean_df = get_average_df(data)
        max_df = abs(mean_df-get_max_df(data))
        min_df = abs(mean_df-get_min_df(data))
        mean_df = get_variation_to_zero(mean_df)
        ax[0].set_title("B_n")
        ax[0].scatter(mean_df['angle'],mean_df['z'], c=COLORS[i], s=12)
        ax[0].errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']],c=COLORS[i], fmt='o', markersize=0)
        ax[0].grid(True)

        ax[1].set_title("B_t")
        ax[1].set_xlabel('Angular Positon [deg]')
        ax[1].scatter(mean_df['angle'],mean_df['y'], c=COLORS[i], marker='s',s=12)
        ax[1].errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']],c=COLORS[i], fmt='s', markersize=0)
        ax[1].grid(True)
        i += 1
    plt.legend(handles=patches, title='Sensing Radius [mm]', loc='lower left')
    plt.tight_layout()
    plt.show()

#Plot X-Component
def sanity_check(data_set):
    patches = []
    i = 0
    for data in data_set:
        patch = mpatches.Patch(color=COLORS[i], label=SENSING[i])
        patches.append(patch)
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
        mean_df = get_average_df(data)
        max_df = abs(mean_df-get_max_df(data))
        min_df = abs(mean_df-get_min_df(data))
        mean_df = get_variation_to_zero(mean_df)
        plt.title("B_x")
        plt.scatter(mean_df['angle'],mean_df['x'], c=COLORS[i], s=12)
        plt.errorbar(mean_df['angle'],mean_df['x'], yerr=[min_df['x'], max_df['x']],c=COLORS[i], fmt='o', markersize=0)
        i += 1
    plt.grid()
    plt.show()

#Optimization Problem of Angle Estimation
# In: Look-Up Table as dataframe + Bn and Bt as float
# Out: Estimatied Angle
def get_angle(data, bn, bt): #capisci bn e bt
    lbnd = 0
    rbnd = 35
    angles = data['angle']
    y_vals = data['y']
    z_vals = data['z']
    wt = 100
    wn = 1
    def minFunction(angle):

        angn_cost = wn*abs(np.interp(angle, angles, z_vals)-bn)
        angt_cost = wt*abs(np.interp(angle, angles, y_vals)-bt)
        print(angle)
        cost = angt_cost+angn_cost
        print(angn_cost, angt_cost)
        return cost
    min_angle = optimize.fminbound(minFunction, 0, 35)
    return min_angle

#Read Values of arduino of format "PPP;x;y;z"
#Out: Bn, Bt as float
def read_arduino(ser):
    is_empty = True
    while is_empty:
        sinput = str(ser.readline().decode('utf-8')).rstrip() #PPP;x;y;z
        if len(sinput) > 0 and sinput.startswith('PPP'):
            is_empty = False
    measure = sinput.split(';')[1:]
    bn = float(measure[2])
    bt = float(measure[1])
    # print('Bn = {}'.format(bn))
    # print('Bt = {}'.format(bt))
    return bn, bt

#Performs error analysis from 0-35° with 3 Cycles 
#In: h_index -> index of height in SENSING/DISTANCES
def error_analyis(ser, h_index):
    degs = np.arange(0, 36, 1)
    error_lst = []
    for n in range(0,3):
        error = []
        angles = []
        bn_lst = []
        bt_lst = []
        for deg in degs:
            ser.write(bytes(str(deg), 'utf-8'))
            bn, bt = read_arduino(ser)
            bn_lst.append(bn)
            bt_lst.append(bt)
            print(bn, bt)
            database = get_data(FILENAME_TEMP, KEYS, DISTANCES[h_index],'male')
            data = pd.concat(database)
            # database = get_multiple_data(FILENAME_TEMP, KEYS, DISTANCES, 'male')
            calculated_angle = get_angle(get_average_df(data), bn, bt)
            angles.append(calculated_angle)
            error.append(calculated_angle-deg)
        df_error = pd.DataFrame({'angle':degs, 'c_angle':angles, 'error': error, 'bn': bn_lst, 'bt':bt_lst})
        error_lst.append(df_error)
    return pd.concat(error_lst)

#Shape Estimation Programm
"""
Choices of entering Angle:
1. "x": close application
2. "error": performs error analysis
3. "live": show live demo of test setup
4. float: desired Angle
"""

def Shape_Estimation():
    ard = serial.Serial(COM, BAUD, timeout=0.1)

    is_active = True
    height = input('Enter Height: ')
    h_index = DISTANCES.index(height)
    # h_index = 1
    while ard.isOpen() is True:
        while is_active:
            servo_angle = input("Enter an Angle: ")
            if servo_angle.lower() == 'x':
                is_active = False
                ard.close()
            elif servo_angle.lower() == 'error':
                type_dist = input("Enter Distance Element: ")
                type_part = input("male/female? ").lower()
                error = error_analyis(ard, h_index)
                error.to_csv(get_filename(FILENAME_TEMP, type_dist, 'ER', type_part))
                print(error.head(12))

            elif servo_angle.lower() == 'live':
                file = get_filename(FILENAME_TEMP, height, 'X', 'male')
                positions = [0,5,10,15,20,25,30,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35, -30, -25, -20, -15, -10, -5]
                k = ['D1V', 'D2V', 'D3V']
                liveplot = AnimationPlot(serial=ard, positions=positions, file_template=file, keys=k)
                ani = animation.FuncAnimation(liveplot.fig, liveplot.animate, frames=100, interval=50)
                # ani.save("line.gif", writer=animation.PillowWriter(fps=20))
                plt.show()

            elif abs(float(servo_angle)) <= 35:
                ard.write(bytes(servo_angle, 'utf-8'))
                bn, bt = read_arduino(ard)
                database = get_multiple_data(FILENAME_TEMP, KEYS, DISTANCES, 'male')
                calculated_angle = get_angle(database[h_index], bn, bt)
                print('Estimated Angle is {:.2f}'.format(calculated_angle))


Shape_Estimation()