# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:11:11 2022

this script is used to decode and plot the data streams coming out of the TMS320 DSP

@author: dvarx
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from numpy.linalg import norm
import numpy as np
from math import sqrt
import os
import serial
import pandas as pd
#import rospkg
#rospack = rospkg.RosPack()
#pkg_path=rospack.get_path("ripple_localization")
from math import pi

#typical resolution of the W2BW sensor, can fluctuate quite a bit between [23.8uT,45.5uT]
tesla_per_bit_akm=1.1e-6

def read_w2bw_tms320_data_syncframe(datasource):
    Bxs=[]
    Bys=[]
    Bzs=[]
    idxs=[]
    if(type(datasource)==bytes):
        idx=0
        while idx + 12 <= len(datasource):
            #check if current index points to start of a frame
            if((datasource[idx]==0xFF and datasource[idx+1]==0xFF and datasource[idx+2]==0xFF)
                    and(datasource[idx+5]==0x00)and(datasource[idx+8]==0x00)and(datasource[idx+11]==0x00)):
                idxs.append(idx)
                #process the current frame
                #parse Bx
                bx_bytes=bytearray([datasource[idx+4],datasource[idx+3]])
                #parse By
                by_bytes=bytearray([datasource[idx+7],datasource[idx+6]])
                #parse Bz
                bz_bytes=bytearray([datasource[idx+10],datasource[idx+9]])
                Bx=tesla_per_bit_akm*int.from_bytes(bx_bytes,"big",signed=True)
                Bxs.append(Bx)
                By=tesla_per_bit_akm*int.from_bytes(by_bytes,"big",signed=True)
                Bys.append(By)
                Bz=tesla_per_bit_akm*int.from_bytes(bz_bytes,"big",signed=True)
                Bzs.append(Bz)
            idx+=1
    #implement parsing of new dataframe here
    return {"Bxs":Bxs,"Bys":Bys,"Bzs":Bzs}


def w2bw_read_n_bytes(N,devfptr="COM12",samplerate=500):
    ser=serial.Serial(devfptr,460800,timeout=N/samplerate*2) #460800
    data_bytes=ser.read(N)
    ser.close()
    return data_bytes

def plot_frame_sig_and_fft(data,freqs=[48,52,56],fs=500,N=500,use_markers=False):
    N0=0
    Bxs_=data[0]
    Bys_=data[1]
    Bzs_=data[2]
    Bxs=np.array(Bxs_[N0:N0+N])
    Bys=np.array(Bys_[N0:N0+N])
    Bzs=np.array(Bzs_[N0:N0+N])
    Bxrms=norm(Bxs)/sqrt(N)
    Byrms=norm(Bys)/sqrt(N)
    Bzrms=norm(Bzs)/sqrt(N)
    
    fig,axs=plt.subplots(3,1)
    axs[0].plot(1e6*Bxs)
    axs[0].set_ylabel("Bx [uT]")
    axs[1].plot(1e6*Bys)
    axs[1].set_ylabel("By [uT]")
    if(use_markers):
        axs[2].plot(1e6*Bzs,"-*")
    else:
        axs[2].plot(1e6*Bzs)
    axs[2].set_ylabel("Bz [uT]")
    
    T=len(Bxs)/fs
    dT=1/fs
    df=1/T
    
    Bxs=np.array(Bxs)
    Bxs_fft=fft(Bxs)[:int(len(Bxs)/2)]  #/2
    Bxs_fft=Bxs_fft/norm(Bxs_fft)*Bxrms
    Bxs_fft=abs(Bxs_fft)

    Bys_fft=fft(Bys)[:int(len(Bys)/2)]
    Bys_fft=Bys_fft/norm(Bys_fft)*Byrms
    Bys_fft=abs(Bys_fft)

    Bzs_fft=fft(Bzs)[:int(len(Bzs)/2)]
    Bzs_fft=Bzs_fft/norm(Bzs_fft)*Bzrms
    Bzs_fft=abs(Bzs_fft)
    
    Bs_fft=np.sqrt(Bxs_fft**2+Bys_fft**2+Bzs_fft**2)
    
    #compute powers at frequencies
    print("Pf0: %f ; Pf1: %f ; Pf2: %f"%(1e6*Bs_fft[int(freqs[0]/df)],1e6*Bs_fft[int(freqs[1]/df)],1e6*Bs_fft[int(freqs[2]/df)]))

    #plot dft of different frequencies
    plt.figure()
    plt.subplot(311)
    plt.plot(np.arange(len(Bs_fft))*df,Bxs_fft,"b-*")
    plt.ylabel("DFT[Bx]")
    plt.xlabel("freq (Hz)")
    plt.subplot(312)
    plt.plot(np.arange(len(Bs_fft))*df,Bys_fft,"b-*")
    plt.ylabel("DFT[By]")
    plt.xlabel("freq (Hz)")
    plt.subplot(313)
    plt.plot(np.arange(len(Bs_fft))*df,Bzs_fft,"b-*")
    plt.ylabel("DFT[Bz]")
    plt.xlabel("freq (Hz)")


    #plot indicators for frequencies 
    plt.figure()
    plt.plot(np.arange(len(Bs_fft))*df,1e6*Bs_fft,"b-*")
    plt.ylabel("$\hat{B}[k]$ [uTrms]")
    plt.xlabel("freq (Hz)")

    maxfft=max(Bs_fft)*1e6
    for freqno in range(0,3):
        plt.plot([freqs[freqno],freqs[freqno]],[0,maxfft],"red",alpha=0.25)

    plt.show()


    # Parameters
    time_step = 1 / fs  # Time step in seconds
    N = len(Bxs)
    # Create time array
    time = np.linspace(0, (N - 1) * time_step, N)  # Time array from 0 to the total duration

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot Bx versus time
    axs[0].plot(time * 1e3, 1e6 * Bxs)  # Time in milliseconds, Bx in microtesla
    axs[0].set_ylabel("Bx [µT]")
    axs[0].set_xlabel("Time [ms]")
    axs[0].grid(True)

    # Plot By versus time
    axs[1].plot(time * 1e3, 1e6 * Bys)  # Time in milliseconds, By in microtesla
    axs[1].set_ylabel("By [µT]")
    axs[1].set_xlabel("Time [ms]")
    axs[1].grid(True)

    # Plot Bz versus time with or without markers
    axs[2].plot(time * 1e3, 1e6 * Bzs)  # Time in milliseconds, Bz in microtesla
    axs[2].set_ylabel("Bz [µT]")
    axs[2].set_xlabel("Time [ms]")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    no=5
    #filename="applications/ripple_localization/data/calibration/100hz_solidcore/background_current_calibration/nonlin_meascurrent_coil_1_31"
    # filename=pkg_path+"/data/ac_gain_meas_rep5.log"
    # #filename="/home/dvarx/src/utilities/lockin_amp/tms320_debug.log"
    # data=read_w2bw_tms320_data(filename,correct_errors=False)
    # Bxs_=data["Bxs"]
    # Bys_=data["Bys"]
    # Bzs_=data["Bzs"]
    # print("read %d field measurements"%(len(Bxs_)))
    #N=len(Bxs)-1

    df = pd.read_csv('CalibrationAC0.csv', index_col='Index')

    Bxs = df['x'].values
    Bys = df['y'].values 
    Bzs = df['z'].values 


    plot_frame_sig_and_fft([Bxs,Bys,Bzs],freqs=[92,98,104],N=len(Bxs),fs=2e3)