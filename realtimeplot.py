import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tesla_per_bit_akm = 1.1e-6

def read_w2bw_tms320_data_syncframe(datasource):
    Bxs = []
    Bys = []
    idx = 0
    while not idx + 10 > len(datasource):
        if (((datasource[idx] == 0xAA and datasource[idx+1] == 0xAA and datasource[idx+2] == 0xAA) or 
             (datasource[idx] == 0xFF and datasource[idx+1] == 0xFF and datasource[idx+2] == 0xFF)) and 
            (datasource[idx+5] == 0x00) and (datasource[idx+8] == 0x00) and (datasource[idx+11] == 0x00)):
            
            bx_bytes = bytearray([datasource[idx+4], datasource[idx+3]])
            by_bytes = bytearray([datasource[idx+7], datasource[idx+6]])
            
            Bx = tesla_per_bit_akm * int.from_bytes(bx_bytes, "big", signed=True)
            By = tesla_per_bit_akm * int.from_bytes(by_bytes, "big", signed=True)
            
            Bxs.append(Bx)
            Bys.append(By)
        idx += 1
    return Bxs, Bys

def w2bw_read_n_bytes(N, devfptr="COM3", samplerate=500):
    ser = serial.Serial(devfptr, 460800, timeout=N/samplerate*2)
    data_bytes = ser.read(N)
    ser.close()
    return data_bytes

def update_plot(i, xs, ys, line1, line2, N, devfptr):
    data_bytes = w2bw_read_n_bytes(N, devfptr)
    Bxs, Bys = read_w2bw_tms320_data_syncframe(data_bytes)
    
    xs.extend(Bxs)
    ys.extend(Bys)
    
    xs = xs[-100:]  # Keep only the last 100 values for plotting
    ys = ys[-100:]
    
    line1.set_ydata(xs)
    line2.set_ydata(ys)
    
    return line1, line2

def main():
    N = 1000  # Number of bytes to read
    new_serial_port = "COM8"  # New serial port to use
    
    fig, ax = plt.subplots()
    xs = [0] * 100  # Initialize with 100 zero values
    ys = [0] * 100  # Initialize with 100 zero values
    
    line1, = ax.plot(xs, label='Bx')
    line2, = ax.plot(ys, label='By')
    
    ax.legend()
    
    ani = animation.FuncAnimation(fig, update_plot, fargs=(xs, ys, line1, line2, N, new_serial_port), interval=100)
    
    plt.show()

if __name__ == "__main__":
    main()
