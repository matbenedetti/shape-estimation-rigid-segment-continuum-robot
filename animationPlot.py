import time
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

class AnimationPlot:
    def __init__(self, serial, positions, file_template, keys):
        self.serial = serial
        self.positions = positions
        self.index_position = 0
        self.keys = keys
        self.file_temp = file_template
        self.look_data = self.get_data(self.file_temp)
        print(self.look_data.head(40))
        self.angles = []
        self.servo_positions = []

        self.colors = plt.cm.viridis(np.linspace(0,1,3))
        self.patches = []
        patch = mpatches.Patch(color=self.colors[0], label='Servo Angle')
        self.patches.append(patch)
        patch = mpatches.Patch(color=self.colors[1], label='Estimated Angle')
        self.patches.append(patch)

        self.fig = plt.figure(figsize=(10,10))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)


    def get_data(self, file):
        data_lst = []
        for key in self.keys:
            file  = file.replace('X', key)
            data_lst.append(pd.read_csv(file, index_col=0))
        df = pd.concat(data_lst)
        rows = list(set(df.index.tolist()))
        average = pd.DataFrame()
        for row in rows:
            m = df.loc[row].mean().to_frame().T
            average = pd.concat([average,m], ignore_index=True)
        return average
    
    def animate(self,i):
        if i > 5:
            pos = self.positions[self.index_position]
            self.servo_positions.append(pos)
            bn, bt = self.write_read_arduino(pos)
            angle = self.get_angle(self.look_data, bn, bt)
            self.angles.append(angle)
            self.angles = self.angles[-20:]
            self.servo_positions = self.servo_positions[-20:]
            x = list(range(len(self.angles)))
            error = self.subtract_list(self.angles,self.servo_positions)
            clrs = ['red' if (x<0) else 'green' for x in error]
            self.ax1.clear()
            self.ax2.clear()
            self.getPlotFormat()
            self.ax1.plot(x,self.angles, color=self.colors[1], linestyle='--', marker='v', label='Estimated Angle')
            self.ax1.plot(x,self.servo_positions, color=self.colors[0], linestyle='-', marker='o', label='Servo Angle')
            self.ax2.bar(x=x,height=error, color=clrs)
            self.next_position()
        else:
            self.getPlotFormat()
            time.sleep(1)
        return self.ax1, self.ax2



    def getPlotFormat(self):
        self.ax1.set_xticks([])
        self.ax2.set_xticks([])
        self.ax1.set_xlim([0,20])
        self.ax1.set_ylim([-45,45])
        self.ax2.set_ylim([-35,35])
        # self.ax1.set_title("Angle Estimation")
        self.ax1.set_ylabel("Angle [°]")
        self.ax2.set_ylabel(r"$\Delta$Angle [°]")
        self.ax1.grid(True)
        self.ax2.grid(True)
        triangle_patch = mpatches.RegularPolygon((0.5, 0.5), numVertices=3, radius=0.1, color=self.colors[1], label=r"Estimated $\alpha^'$")
        circle_patch = mpatches.Circle((0.5, 0.5), radius=0.1, color=self.colors[0], label=r'$\alpha$')
        self.ax1.legend(handles=[triangle_patch, circle_patch])
        self.ax2.set_xlim([0,20])
        # self.ax2.set_ylim([-40,40])

    def write_read_arduino(self, input_angle):
        input_angle = str(input_angle)
        self.serial.write(bytes(input_angle, 'utf-8'))
        is_empty = True
        while is_empty:
            ard_input =  str(self.serial.readline().decode('utf-8').rstrip()) ##PPP;x;y;z
            if len(ard_input) > 0 and ard_input.startswith('PPP'):
                is_empty = False
        measurement = ard_input.split(';')[1:]
        bt = float(measurement[1])
        bn = float(measurement[2])
        return bn, bt
    
    def next_position(self):
        if self.index_position < len(self.positions)-1:
            self.index_position += 1
        else:
            self.index_position = 0
    
    def get_angle(self,data, bn, bt):
        lbnd = -35
        rbnd = 35
        angles = data['angle']
        y_vals = data['y']
        z_vals = data['z']
        wt = 10
        wn = 1
        def minFunction(angle):
            angn_cost = wn*abs(np.interp(angle, angles, z_vals)-bn)
            angt_cost = wt*abs(np.interp(angle, angles, y_vals)-bt)
            cost = angt_cost+angn_cost
            return cost
        min_angle = optimize.fminbound(minFunction, -35, 35)
        return min_angle
    
    def subtract_list(self, list1, list2):
        return [float(a) - float(b) for a,b in zip(list1,list2)]