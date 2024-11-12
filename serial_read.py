import serial
import time
import csv
import pandas as pd

"""
IMPORTANT:

Load Shape_Estimation_Test_Setup_Angle_Micro to Arduino


This File is used to gather the Data of Magnetic Field
"""



com = "COM3"
baud = 2000000

cols = ["x", "y", "z", "angle"]

x = serial.Serial(com, baud, timeout=0.2)
data_dic = {"1": pd.DataFrame(columns=cols), "2": pd.DataFrame(columns=cols), "3":pd.DataFrame(columns=cols)}
#data_dic = []

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

key = ""
while x.isOpen() is True:
    data = str(x.readline().decode('utf-8')).rstrip()
    print(data)
    if len(data) == 1:
        key = data
        continue
    elif has_numbers(data) and data != "":
        data_arr = pd.DataFrame([[float(item) for item in data.split(',')]], columns=cols)
        data_dic[key] = pd.concat([data_dic[key], data_arr], ignore_index=True)
    elif data == "XXX":
        x.close()
        break
    else:
        continue
print(data_dic["1"])
print(data_dic["2"])


#Modify Name of Data
data_dic["1"].to_csv("R1_male_0.2A_200_11.85mm")
data_dic["2"].to_csv("R2_male_0.2A_200_11.85mm")
data_dic["3"].to_csv("R3_male_0.2A_200_11.85mm")