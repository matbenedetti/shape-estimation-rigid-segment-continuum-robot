import serial
import struct
import time
import sys
from numpy import array, double


class sutterMP285:
    'Class which allows interaction with the Sutter Manipulator 285'

    def __init__(self):
        self.verbose = 1.  # level of messages
        self.timeOut = 30  # timeout in sec
        # initialize serial connection to controller
        try:
            self.ser = serial.Serial(port='/dev/ttyS0', baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=self.timeOut)
            self.connected = 1
            if self.verbose:
                print(self.ser)
        except serial.SerialException:
            print('No connection to Sutter MP-285 could be established!')
            sys.exit(1)

        # set move velocity to 200
        self.setVelocity(200, 10)
        self.updatePanel()  # update controller panel
        (stepM, currentV, vScaleF) = self.getStatus()
        if currentV == 200:
            print('sutterMP285 ready')
        else:
            print('sutterMP285: WARNING Sutter did not respond at startup.')

    # destructor
    def __del__(self):
        self.ser.close()
        if self.verbose:
            print('Connection to Sutter MP-285 closed')

    def getPosition(self):
        # send command to get position
        self.ser.write(b'c\r')
        # read position from controller
        xyzb = self.ser.read(13)
        # convert bytes into 'signed long' numbers
        xyz_um = array(struct.unpack('iii', xyzb[:12])) / self.stepMult

        if self.verbose:
            print('sutterMP285 : Stage position ')
            print('X: %g um \nY: %g um\nZ: %g um' % (xyz_um[0], xyz_um[1], xyz_um[2]))

        return xyz_um

    # Moves the three axes to specified location.
    def gotoPosition(self, pos):
        if len(pos) != 3:
            print('Length of position argument has to be three')
            sys.exit(1)
        xyzb = struct.pack('lll', int(pos[0] * self.stepMult), int(pos[1] * self.stepMult), int(pos[2] * self.stepMult))  # convert integer values into bytes
        startt = time.time()  # start timer
        self.ser.write(b'm' + xyzb + b'\r')  # send position to controller; add the "m" and the CR to create the move command
        cr = self.ser.read(1)  # read carriage return and ignore
        endt = time.time()  # stop timer
        if len(cr) == 0:
            print('Sutter did not finish moving before timeout (%d sec).' % self.timeOut)
        else:
            print('sutterMP285: Sutter move completed in (%.2f sec)' % (endt - startt))

    # this function changes the velocity of the sutter motions
    def setVelocity(self, Vel, vScalF=10):
        # Change velocity command 'V'xxCR where xx = unsigned short (16bit) int velocity
        # set by bits 14 to 0, and bit 15 indicates ustep resolution  0 = 10, 1 = 50 uSteps/step
        # V is ascii 86
        # convert velocity into unsigned short - 2-byte - integer
        velb = struct.pack('H', int(Vel))
        # change last bit of 2nd byte to 1 for ustep resolution = 50
        if vScalF == 50:
            velb2 = double(struct.unpack('B', velb[1:2])[0]) + 128
            velb = velb[0:1] + struct.pack('B', int(velb2))
        self.ser.write(b'V' + velb + b'\r')
        self.ser.read(1)

    # Update Panel
    # causes the Sutter to display the XYZ info on the front panel
    def updatePanel(self):
        self.ser.write(b'n\r')  # Sutter replies with a CR
        self.ser.read(1)  # read and ignore the carriage return

    ## Set Origin
    # sets the origin of the coordinate system to the current position
    def setOrigin(self):
        self.ser.write(b'o\r')  # Sutter replies with a CR
        self.ser.read(1)  # read and ignore the carriage return

    # Reset controller
    def sendReset(self):
        self.ser.write(b'r\r')  # Sutter does not reply

    # Queries the status of the controller.
    def getStatus(self):
        if self.verbose:
            print('sutterMP285: get status info')
        self.ser.write(b's\r')  # send status command
        rrr = self.ser.read(32)  # read return of 32 bytes without carriage return
        self.ser.read(1)  # read and ignore the carriage return
        statusbytes = struct.unpack(32 * 'B', rrr)
        print(statusbytes)
        # the value of STEP_MUL ("Multiplier yields msteps/nm") is at bytes 25 & 26
        self.stepMult = double(statusbytes[25]) * 256 + double(statusbytes[24])

        # the value of "XSPEED"  and scale factor is at bytes 29 & 30
        if statusbytes[29] > 127:
            self.vScaleFactor = 50
        else:
            self.vScaleFactor = 10
        self.currentVelocity = double(127 & statusbytes[29]) * 256 + double(statusbytes[28])

        if self.verbose:
            print('step_mul (usteps/um): %g' % self.stepMult)
            print('xspeed" [velocity] (usteps/sec): %g' % self.currentVelocity)
            print('velocity scale factor (usteps/step): %g' % self.vScaleFactor)

        return (self.stepMult, self.currentVelocity, self.vScaleFactor)
