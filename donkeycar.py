import donkeycar as dk
from donkeycar.parts import gps as g
from donkeycar.parts import gps1 as g1
import serial

#initialize the vehicle
V = dk.Vehicle()
#s = serial.Serial('COM3')

serialPort = serial.Serial(
    port="COM3", baudrate=9600, bytesize=8, timeout=1, stopbits=serial.STOPBITS_ONE
)
serialString = ""  # Used to hold data coming over UART
while 1:
    # Wait until there is data waiting in the serial buffer
    if serialPort.in_waiting > 0:

        # Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()
        # Print the contents of the serial data
        try:
            print(serialString.decode("Ascii"))
            #if "blah" not in somestring: 
        except:
            pass

gps = g.Gps("COM3", 9600, .5, False)
#gps = g1.GPS(9600, "COM3", .5)

print(gps.run())
#while(1):
    #print(gps.run_threaded())