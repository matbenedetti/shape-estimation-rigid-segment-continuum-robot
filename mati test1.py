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
DISTANCES = ['11.0', '11.15', '11.30', '11.5', '11.70', '11.85', '12.0'] #Height element of Test Setup

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


# Get data
# Out: List of all keys and distances as data frame
def get_multiple_data(file_temp, keys, distances, part):
    data_lst = []
    for distance in distances:
        all_data = get_data(FILENAME_TEMP, KEYS, distance, part)
        data = pd.concat(all_data)
        data_lst.append(data)
    return data_lst

#Optimization Problem of Angle Estimation
# In: Look-Up Table as dataframe + Bn and Bt as float
# Out: Estimatied Angle
def get_angle(data, bn, bt, distances):
    lbnd = 0
    rbnd = 35
    wt = 100
    wn = 1
    min_angle = None
    min_cost = float('inf')
    min_dis = None

    for dis in distances:
        data_dis = data[dis]

        def minFunction(angle):
            angles = data_dis['angle']
            y_vals = data_dis['y']
            z_vals = data_dis['z']
            angn_cost = wn * abs(np.interp(angle, angles, z_vals) - bn)
            angt_cost = wt * abs(np.interp(angle, angles, y_vals) - bt)
            cost = angt_cost + angn_cost
            return cost

        result = optimize.fminbound(minFunction, lbnd, rbnd)
        angle = result.x
        cost = result.fun

        if cost < min_cost:
            min_cost = cost
            min_angle = angle
            min_dis = dis

    return min_angle, min_dis

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