import time
import struct
import smbus2
import serial

# I2C Device Address
DEVICE_ADDRESS = 0x6A  # Device address for version A0

# Register Addresses
ADDR_CONFIG = 0x10
ADDR_MOD = 0x11
ADDR_CONFIG2 = 0x14

# Configuration bytes
WRITE_CONFIG_CONFIG_REG = [ADDR_CONFIG, 0x20 | 0x08 | 0x01]
WRITE_CONFIG_MOD1_REG = [ADDR_MOD, 0x01 | 0x10 | 0x80]
WRITE_CONFIG_CONFIG2_REG = [ADDR_CONFIG2, 0x01]

# Measurement structure
class Measurement:
    def __init__(self, Bx=0, By=0, Bz=0, Temp=0):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Temp = Temp

def parse_w2bw_meas(data):
    Bx = (data[0] << 4) | ((data[4] >> 4) & 0x0F)
    By = (data[1] << 4) | (data[4] & 0x0F)
    Bz = (data[2] << 4) | (data[5] & 0x0F)
    Temp = (data[3] << 4) | ((data[5] >> 6) & 0x03)
    return Measurement(Bx, By, Bz, Temp)

def serialize_w2bw(meas):
    return struct.pack('<HHHH', meas.Bx, meas.By, meas.Bz, meas.Temp)

def main():
    # Initialize I2C bus and UART
    i2c_bus = smbus2.SMBus(1)
    uart = serial.Serial('/dev/ttyUSB0', 460800, timeout=1)  # Adjust the serial port as needed

    # Initialize the sensor
    time.sleep(0.2)
    i2c_bus.write_i2c_block_data(DEVICE_ADDRESS, WRITE_CONFIG_CONFIG_REG[0], WRITE_CONFIG_CONFIG_REG[1:])
    i2c_bus.write_i2c_block_data(DEVICE_ADDRESS, WRITE_CONFIG_MOD1_REG[0], WRITE_CONFIG_MOD1_REG[1:])
    i2c_bus.write_i2c_block_data(DEVICE_ADDRESS, WRITE_CONFIG_CONFIG2_REG[0], WRITE_CONFIG_CONFIG2_REG[1:])

    while True:
        # Read data from I2C sensor
        data = i2c_bus.read_i2c_block_data(DEVICE_ADDRESS, 0, 7)
        meas = parse_w2bw_meas(data)

        # Serialize and send data via UART
        serialized_data = serialize_w2bw(meas)
        uart.write(serialized_data)

        # Optional: add delay for timing control
        time.sleep(1)

if __name__ == '__main__':
    main()
