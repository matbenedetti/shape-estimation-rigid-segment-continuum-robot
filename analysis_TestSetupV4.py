import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from math import sqrt

'''
This File is used to analyze data. Load all files with "key" name in "KEYS" to the same folder
'''


FILENAME_TEMP = "X_?_0.2A_200_Ymm" #R2_male_0.2A_200_13.0mm
# DISTANCES = ['10.5', '11.5', '12.5', '13.0']
# SENSING = ['4.5', '3.5', '2.5', '2']


#SENSING and DISTANCES must have same len()
DISTANCES = ['10.5', '11.0', '11.5', '12.0', '12.5', '13.0']
DISTANCES = ['10.5', '11.0', '11.5', '12.0']
SENSING = ['4.5', '4.0', '3.5', '3.0', '2.5', '2']
SENSING = ['4.5', '4.0', '3.5', '3.0']


#Number of Measurement and all available Parts
KEYS = ['R1', 'R2', 'R3']
PARTS = ['male', 'female']

COLORS = plt.cm.viridis(np.linspace(0,1,7))

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


'''
Following functions are used to plot data
'''
def plot_time(dataset):
    fig, ax = plt.subplots(2)
    # fig.suptitle('B-Field Variation to Inital Value of Test Setup V4: 0.2A, 200 Windings, male/female/PLA')
    fig.supylabel('Magnetix Flux Density [$\mu$T]')
    fig.suptitle('B-Field Variation to Inital Value of Testsetup: 0.2A, 200 Windings, male, 4.0mm, Timegap: ~16h')
    patches = []
    i = 0
    for data in dataset:
        patch = mpatches.Patch(color=COLORS[i], label=SENSING[i])
        patches.append(patch)
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
        mean_df = get_average_df(data)
        max_df = abs(mean_df-get_max_df(data))
        min_df = abs(mean_df-get_min_df(data))
        mean_df = get_variation_to_zero(mean_df)
        ax[0].set_title("$\mathregular{B_n}$")
        ax[0].scatter(mean_df['angle'],mean_df['z'], c=COLORS[i], s=12)
        ax[0].errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']],c=COLORS[i], fmt='o', markersize=0)
        ax[0].grid(True)

        ax[1].set_title("$\mathregular{B_t}$")
        ax[1].set_xlabel('Angular Positon [deg]')
        ax[1].scatter(mean_df['angle'],mean_df['y'], c=COLORS[i],s=12)
        ax[1].errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']],c=COLORS[i], fmt='s', markersize=0)
        ax[1].grid(True)
        i += 1
    plt.legend(['Before', 'After 16h'])
    plt.tight_layout()
    plt.show()

def plot_multiple(data_set1, data_set2, data_pla, show_female, show_pla):
    fig, ax = plt.subplots(3,2, sharey='row')
    # fig.suptitle('B-Field Variation to Inital Value of Test Setup V4: 0.2A, 200 Windings, male/female/PLA')
    # fig.supylabel('Magnetic Field[$\mu$T]')
    fig.supxlabel(r'$\alpha$ [°]')
    patches = []
    i = 0
    
    for data in data_set1:
        patch = mpatches.Patch(color=COLORS[i], label=SENSING[i])
        patches.append(patch)
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
        mean_df = get_average_df(data)
        max_df = abs(mean_df-get_max_df(data))
        min_df = abs(mean_df-get_min_df(data))
        mean_df = get_variation_to_zero(mean_df)
        ax[0,0].set_title("$\mathregular{B_z}$-$\mathregular{B_{z,0}}$",fontsize=20)
        ax[0,0].scatter(mean_df['angle'],mean_df['z'], c=COLORS[i], s=12)
        ax[0,0].errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']],c=COLORS[i], fmt='o', markersize=0)
        ax[0,0].grid(True)

        ax[1,0].set_title("$\mathregular{B_y}$-$\mathregular{B_{y,0}}$",fontsize=20)
        # ax[1,0].set_xlabel('male')
        ax[1,0].scatter(mean_df['angle'],mean_df['y'], c=COLORS[i],s=12)
        ax[1,0].errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']],c=COLORS[i], fmt='s', markersize=0)
        ax[1,0].grid(True)

        ax[2,0].set_title("$\mathregular{B_x}$-$\mathregular{B_{x,0}}$")
        ax[2,0].set_xlabel('male', fontsize=20)
        ax[2,0].scatter(mean_df['angle'],mean_df['x'], c=COLORS[i],s=12)
        ax[2,0].errorbar(mean_df['angle'],mean_df['x'], yerr=[min_df['x'], max_df['x']],c=COLORS[i], fmt='s', markersize=0)
        ax[2,0].grid(True)
        i += 1
    ax[0,0].set_ylabel("Magnetic Field[$\mu$T]")
    ax[1,0].set_ylabel("Magnetic Field[$\mu$T]")
    ax[2,0].set_ylabel("Magnetic Field[$\mu$T]")
    
    if show_female:
        i = 0
        custom_markers = ''
        for data in data_set2:
            data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
            mean_df = get_average_df(data)
            max_df = abs(mean_df-get_max_df(data))
            min_df = abs(mean_df-get_min_df(data))
            mean_df = get_variation_to_zero(mean_df)
            ax[0,1].set_title("$\mathregular{B_z}$-$\mathregular{B_{z,0}}$", fontsize = 20)
            ax[0,1].scatter(mean_df['angle'],mean_df['z'], c=COLORS[i],marker='v', s=12)
            ax[0,1].errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']],c=COLORS[i], fmt='P', markersize=0)
            ax[0,1].grid(True)

            ax[1,1].set_title("$\mathregular{B_y}$-$\mathregular{B_{y,0}}$", fontsize = 20)
            ax[1,1].scatter(mean_df['angle'],mean_df['y'], c=COLORS[i], marker='v',s=12)
            ax[1,1].errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']],c=COLORS[i], fmt='P', markersize=0)
            ax[1,1].grid(True)
            ax[2,1].set_xlabel('female')

            ax[2,1].set_title("$\mathregular{B_x}$-$\mathregular{B_{x,0}}$", fontsize = 20)
            ax[2,1].scatter(mean_df['angle'],mean_df['x'], c=COLORS[i],marker='v', s=12)
            ax[2,1].errorbar(mean_df['angle'],mean_df['x'], yerr=[min_df['x'], max_df['x']],c=COLORS[i], fmt='P', markersize=0)
            ax[2,1].grid(True)

            i += 1


    if show_pla:
            data = pla
            color_pla = 6
            data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
            mean_df = get_average_df(data)
            max_df = abs(mean_df-get_max_df(data))
            min_df = abs(mean_df-get_min_df(data))
            mean_df = get_variation_to_zero(mean_df)
            ax[0,0].scatter(mean_df['angle'],mean_df['z'], c=COLORS[color_pla],marker='s', s=12)
            ax[0,0].errorbar(mean_df['angle'],mean_df['z'], yerr=[min_df['z'], max_df['z']],c=COLORS[color_pla], fmt='P', markersize=0)

            ax[1,0].scatter(mean_df['angle'],mean_df['y'], c=COLORS[color_pla], marker='s',s=12)
            ax[1,0].errorbar(mean_df['angle'],mean_df['y'], yerr=[min_df['y'], max_df['y']],c=COLORS[color_pla], fmt='P', markersize=0)

            ax[2,0].scatter(mean_df['angle'],mean_df['x'], c=COLORS[color_pla], marker='s',s=12)
            ax[2,0].errorbar(mean_df['angle'],mean_df['x'], yerr=[min_df['x'], max_df['x']],c=COLORS[color_pla], fmt='P', markersize=0)

            patch_PLA = mpatches.Patch(color=COLORS[color_pla], label='3 (PLA)')
            patches.append(patch_PLA)

    # custom_markers = [
    # Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=10),
    # Line2D([0], [0], marker='v', color='w', markerfacecolor='k', markersize=10),
    # Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS[color_pla], markersize=10)
    # ]
    # ax[0,0].legend(handles=custom_markers, labels=['male','female', 'PLA'], loc='upper left')
    ax[1,0].legend(handles=patches, title='Sensing Radius [mm]', loc='lower left')
    plt.tight_layout()
    plt.show()

def show_magnetization():
    mag1 = get_data(FILENAME_TEMP, ['M1', 'M2', 'M3'], '12.5', 'male')
    mag2 = get_data(FILENAME_TEMP, ['M4', 'M5', 'M6'], '12.5', 'male')

    fig, ax = plt.subplots(2)
    fig.suptitle('B-Field Variation to Inital Value of Test Setup V4: 0.2A, 200 Windings, male')
    fig.supylabel('Magnetix Flux Density [uT]')
    patches = []
    ROUND = ['Initial', 'Magnetized', 'OFF']

    i = 0
    for data in mag2:
        patch = mpatches.Patch(color=COLORS[i], label=ROUND[i])
        patches.append(patch)
        data[['x', 'y', 'z']] = data[['x', 'y', 'z']].apply(lambda x: x*-1)
        mean_df = get_variation_to_zero(data)
        ax[0].scatter(mean_df['angle'],mean_df['z'], c=COLORS[i],marker='o', s=12)

        ax[1].scatter(mean_df['angle'],mean_df['y'], c=COLORS[i], marker='o',s=12)
        i += 1

    ax[1].set_title("B_t")
    ax[0].set_title("B_n")
    
    ax[1].grid(True)
    ax[0].grid(True)
    plt.legend(handles=patches, title='Phase of Cycle', loc='lower left')
    plt.tight_layout()
    plt.show()

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
    
def plot_error(dataset):
    fig, ax = plt.subplots(1,2)
    #ax[0].set_title("Angle Estimation Accuracy")
    i = 0
    patches = []
    means = []
    RMSE = []
    for data in dataset:
        patch = mpatches.Patch(color=COLORS[i], label=SENSING[i])
        patches.append(patch)
        mean_df = get_average_df(data)
        max_df = abs(mean_df-get_max_df(data))
        min_df = abs(mean_df-get_min_df(data))
        # print(mean_df.keys())
        RSME_df = sqrt(mean_squared_error(data['angle'], data['c_angle']))
        RMSE.append(RSME_df)
        
        # RSME_df = data['c_angle'].std()
        mean = abs(data['error']).mean()
        means.append(mean)
        # print("Average Error: {}".format(mean_er_df))
        # print("Average max Error: {}".format(mean_er_df+max_er_df))
        # print("Average min Error: {}".format(mean_er_df-min_er_df))
        # ax[0].set_ylabel(r"Estimated $\alpha^'$ [°]")
        # ax[0].set_xlabel(r"$\alpha$ [°]")
        ax[0].scatter(mean_df['angle'], mean_df['c_angle'], s=12, c=COLORS[i])
        ax[0].errorbar(mean_df['angle'], mean_df['c_angle'], yerr=[min_df['c_angle'], max_df['c_angle']], c=COLORS[i], fmt = 'o', markersize=0)

        # ax[1].set_xlabel("Sensing radius [mm]")
        # ax[1].set_ylabel("Estimation error [°]")
        ax[1].scatter(float(SENSING[i]), mean, s=12, c=COLORS[i])
        ax[1].errorbar(float(SENSING[i]), mean, yerr=RSME_df, c=COLORS[i], capsize=4, fmt = 'o', markersize=0)

        i += 1
    ax[0].plot(mean_df['angle'] ,mean_df['angle'], linestyle='dashed', label='ideal')
    ax[0].legend(handles=patches, title='Sensing Radius [mm]', loc='upper left')
    ax[0].grid(True)
    ax[1].grid(axis='y')
    print(RMSE)
    print(means)
    plt.show()
    
def angle_mag():
    data = pd.read_csv('A_35_200_male.csv')
    fig, ax = plt.subplots(2)
    fig.suptitle('B-Field Variation to Inital Value at 35° of Simulation: 0.2A, 200 Windings, male')
    var = get_variation_to_zero(data)
    var['m'] = data['m']

    fig.supylabel('Magnetix Flux Density [$\mu$T]')
    ax[0].set_title("$\mathregular{B_n}$")
    ax[0].scatter(var['m']*1000,var['bn']*10**6*-1, c=COLORS[0], s=12)
    ax[0].grid(True)

    ax[1].set_title("$\mathregular{B_t}$")
    ax[1].scatter(var['m']*1000,var['bt']*10**6*-1, c=COLORS[1], s=12)
    ax[1].grid(True)
    ax[1].set_xlabel('Sensing Radius [mm]')
    plt.tight_layout()
    plt.show()

def plot_matlab():
    SCALE = 10**6
    LEGEND = ['2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
    COMP = ['mbx', 'mby', 'mbz']
    FOLDER = "../Matlab_3D_MagFlux/"
    # bx = pd.read_csv("../Matlab_3D_MagFlux/mbx.csv")*SCALE
    by = pd.read_csv("mby.csv",header=None)*SCALE
    bz = pd.read_csv("mbz.csv", header=None)*SCALE
    bnorm = pd.read_csv("mbnorm.csv", header=None)*SCALE
    deg = pd.read_csv("deg.csv", header=None)

    bz = bz.loc[:,0:389]
    by = by.loc[:,0:389]
    deg = deg.loc[:,0:389]
    bnorm = bnorm.loc[:,0:389]

    rows = bz.shape[0]
    fig, ax = plt.subplots(3)
    # fig.supylabel('Magnetic Field [$\mu$T]')
    # fig.supxlabel('$\alpha$ [°]')
    patches = []
    for row in range(0,1):
        rowc = row
        row = rows-row-1
        name = LEGEND[row]
        patch = mpatches.Patch(color=COLORS[rowc], label=name)
        patches.append(patch)
        ax[0].plot(deg.loc[0], bnorm.loc[row, :], c=COLORS[rowc])
        ax[0].set_title("$\mathregular{||B||_2}$")
        ax[1].plot(deg.loc[0], bz.loc[row, :], c=COLORS[rowc])
        ax[1].set_title("$\mathregular{B_z}$", fontsize=20)
        ax[2].plot(deg.loc[0], by.loc[row, :], c=COLORS[rowc])
        ax[2].set_title("$\mathregular{B_y}$", fontsize=20)
    # ax[0].set_ylabel("Magnetic Field [$\mu$T]")
    # ax[1].set_ylabel("Magnetic Field [$\mu$T]", fontsize=20)
    # ax[2].set_ylabel("Magnetic Field [$\mu$T]", fontsize=20)
    ax[1].set_ylim(1300, 2300)
    ax[2].set_ylim(-1800, 0)

    ax[2].set_xlabel(r"$\alpha$ [°]", fontsize=20)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[0].axis('off')
    ax[2].legend(handles=patches, title='Sensing Radius [mm]', loc='lower left')
    plt.show()

def plot_2D_material():
    SCALE = 10**6
    CORS = plt.cm.viridis(np.linspace(0,1,4))
    column_names = ['bn', 'bt', 'angle']
    mu0 = pd.read_csv('2D_mu0.csv', header=None, names=column_names)
    mu62 = pd.read_csv('2D_mu62.csv', header=None, names=column_names)
    mu1800 = pd.read_csv('2D_mu1800.csv', header=None,names=column_names)
    data_set = [mu0, mu62, mu1800]
    name = ['PLA', 'Steel type 440B', 'Steel type 440F']
    fig, ax = plt.subplots(2)
    # fig.supylabel('Magnetic Field [$\mu$T]')
    fig.supxlabel('Angular Positon [°]')
    patches = []
    i = 0
    for data in data_set:
        patch = mpatches.Patch(color=CORS[i], label=name[i])
        patches.append(patch)
        ax[0].plot(data['angle'], data['bn']*SCALE, c=CORS[i])
        ax[0].set_ylabel("Magnetic Field [$\mu$T]")
        ax[1].set_ylabel("Magnetic Field [$\mu$T]")
        ax[0].set_title("$\mathregular{B_z}$",fontsize=14)
        ax[1].plot(data['angle'], data['bt']*SCALE, c=CORS[i])
        ax[1].set_title("$\mathregular{B_y}$",fontsize=14)
        i+=1
    ax[0].grid(True)
    ax[1].grid(True)
    ax[1].legend(handles=patches, title='Material', loc='lower left')
    plt.show()

def transform_to_error(df, keys):
    for key in keys:
        df[key] = df['angle']-df[key]
    return df

def RMSE_lst(df, keys):
    #only for plot_3d_error()
    RMSE = []
    for key in keys:
        rm = mean_squared_error(df['angle'], df[key], squared=False)
        RMSE.append(rm)
    return RMSE

def MEAN_lst(df, keys):
    MEAN = []
    for key in keys:
        me = abs(df[key]).mean()
        MEAN.append(me)
    return MEAN

def plot_3D_error():
    width = 0.05
    col_names = ['angle','2', '2.5', '3','3.5', '4', '4.5', '5']
    dist = ['2', '2.5', '3','3.5', '4', '4.5', '5']
    fdist = [float(item) for item in dist]
    fdist1 = [float(item)-width for item in dist]
    fdist2 = [float(item)+width for item in dist]

    fdist = [fdist1, fdist2]

    se1 = pd.read_csv('3D_calc_0.15.csv', header=None, names=col_names)
    se1 = se1.loc[0:195, :]
    se2 = pd.read_csv('3D_calc_1.1.csv', header=None, names=col_names)
    se2 = se2.loc[0:195, :]
    se1_error = transform_to_error(se1.copy(), dist)
    se2_error = transform_to_error(se2.copy(), dist)

    print(se1_error['2'].max())
    se1_mean = MEAN_lst(se1_error,dist)
    print(min(se1_mean))
    se2_mean = MEAN_lst(se2_error,dist)
    se1_rs = RMSE_lst(se1, dist)
    print(min(se1_rs))
    se2_rs = RMSE_lst(se2, dist)

    means = [se1_mean, se2_mean]
    rmse = [se1_rs, se2_rs]
    print('Mean S1: ', means[0])
    print('Mean S2: ', means[1])
    print('RMSE S1: ', rmse[0])
    print('RMSE S2: ', rmse[1])
    name = ['0.15', '1.1']
    fig = plt.subplot()
    patches = []
    CORS = plt.cm.viridis(np.linspace(0,1,3))
    for i in range(0, 2):
        mean = means[i]
        rms = rmse[i]
        x = fdist[i]
        patch = mpatches.Patch(color=CORS[i], label=name[i])
        patches.append(patch)
        plt.scatter(x, mean, color=CORS[i])
        plt.errorbar(x, mean, yerr=rms, c=CORS[i], fmt = 'o', markersize=0, capsize=4)

    
    plt.xlabel('Sensing radius [mm]', fontsize=15)
    plt.ylabel('Estimation error [°]', fontsize=15)
    plt.grid(axis='y')

    plt.legend(handles=patches, title='Sensor resolution [uT/LSB]', loc='upper left')
    plt.show()
    
def plot_comp_error():
    tv_mean = []
    tv_rs = []
    for distance in DISTANCES:
        df = pd.read_csv('ER_male_0.2A_200_{}mm'.format(distance), index_col=0)
        tmean = abs(df['error']).mean()
        trs = mean_squared_error(df['angle'], df['c_angle'], squared=False)
        tv_rs.append(trs)
        tv_mean.append(tmean)
    tv_mean = tv_mean[::-1]
    tv_rs = tv_rs[::-1]
    print(min(tv_mean))
    print(tv_rs)

    width = 0.05
    col_names = ['angle','2', '2.5', '3','3.5', '4', '4.5', '5']
    dist = ['2', '2.5', '3','3.5', '4', '4.5']
    fdist = [float(item) for item in dist]
    fdist1 = [float(item)-width for item in dist]
    fdist2 = [float(item)+width for item in dist]

    fdist = [fdist1, fdist,  fdist2]

    se1 = pd.read_csv('3D_calc_0.15.csv', header=None, names=col_names)
    se1 = se1.loc[0:195, :]
    se2 = pd.read_csv('3D_calc_1.1.csv', header=None, names=col_names)
    se2 = se2.loc[0:195, :]
    se1_error = transform_to_error(se1.copy(), dist)
    se2_error = transform_to_error(se2.copy(), dist)

    se1_mean = MEAN_lst(se1_error,dist)
    se2_mean = MEAN_lst(se2_error,dist)
    se1_rs = RMSE_lst(se1, dist)
    se2_rs = RMSE_lst(se2, dist)

    # means = [se1_mean, tv_mean]
    # rmse = [se1_rs, tv_rs]
    # name = ['0.15', 'Test setup']

    means = [se1_mean, tv_mean, se2_mean]
    rmse = [se1_rs, tv_rs, se2_rs]
    name = ['0.15 (Simulation)', '0.15 (Test Setup)', '1.1']
    fig = plt.subplot()
    patches = []
    CORS = plt.cm.viridis(np.linspace(0,1,4))
    for i in range(0, 2):
        mean = means[i]
        rms = rmse[i]
        x = fdist[i]
        patch = mpatches.Patch(color=CORS[i], label=name[i])
        patches.append(patch)
        plt.scatter(x, mean, color=CORS[i])
        plt.errorbar(x, mean, yerr=rms, c=CORS[i], fmt = 'o', markersize=0, capsize=4)


    
    plt.xlabel('Sensing radius [mm]', fontsize=15)
    plt.ylabel('Estimation error [°]', fontsize=15)
    plt.grid(axis='y')

    plt.legend(handles=patches, title='Sensor resolution [uT/LSB]', loc='upper left')
    plt.show()

#Write true, if plot should be shown
show_single = False #Show plot of one single distance - modify below
show_multiple = True #Show plot of all distances - modify below
is_sanity = False #Show X-Component
show_error = False #Show Error of Estimation - modify below
show_time = False #Show time dependency
show_angle_mag = False #Show plot at specific angle

if True:
    if show_single:
        k = KEYS
        k = ['D1', 'D2', 'D3']
        x = get_data(FILENAME_TEMP, k, '11.0', 'male')
        y = pd.concat(x)
        plot_single(y)
    if show_multiple:
        x_m = get_multiple_data(FILENAME_TEMP, KEYS, DISTANCES, 'male')
        x_f = get_multiple_data(FILENAME_TEMP, KEYS, DISTANCES, 'female')
        # x_f = []
        pla = get_data("X_?_0.2A_200_Y", KEYS, 'PLA', 'female')
        pla = pd.concat(pla)
        plot_multiple(x_m, x_f, pla, True, False)

    if is_sanity:
        x = get_multiple_data(FILENAME_TEMP, KEYS, DISTANCES, 'male')
        sanity_check(x)

    if show_error:
        # x = get_multiple_data("X_?_0.2A_200_Ymm", 'ER', DISTANCES, 'male')
        data = []
        for distance in DISTANCES:
            df = pd.read_csv('ER_male_0.2A_200_{}mm'.format(distance), index_col=0)
            data.append(df)
        # x = pd.read_csv('ER_male_0.2A_200_11.0mm', index_col=0)
        # y = pd.read_csv('ERV2_male_0.2A_200_11.0mm', index_col=0)
        # print(x[0].head(3))
        plot_error(data)

    if show_time:
        x = get_multiple_data(FILENAME_TEMP, KEYS,['11.0', '11V2'], 'male')
        plot_time(x)
    #show_magnetization()

    if show_angle_mag:
        angle_mag()

"""
Uncomment if you need
"""
# Show plot of matlab
# plot_matlab()

# Show simulated error estimation
# plot_3D_error()

# Show comparison of error estimation
# plot_comp_error()

# Show plot for material comparison
# plot_2D_material()