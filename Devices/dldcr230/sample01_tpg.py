###########################################################################################################################################################################
# Texas Instruments DLP LightCrafter 230NP EVM Python Support Code - sample01_tpg.py - Last Updated June 4th, 2020
# This script is intended to be used with the DLPDLCR230NP EVM on a Raspberry Pi 4 (or compatible)
# It is recommended to consult the DLPDLCR230NPEVM User's Guide for more detailed information on the operation of this EVM
# For technical support beyond what is provided in the above documentation, direct inquiries to TI E2E Forums (http://e2e.ti.com/support/dlp)
# 
# Copyright (C) 2020 Texas Instruments Incorporated - http://www.ti.com/ 
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met: 
# 
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.
# 
# Neither the name of Texas Instruments Incorporated nor the names of its contributors may be used to endorse or promote products derived from this software 
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###########################################################################################################################################################################
import struct
import time

from enum import Enum

import sys, os.path
python_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(python_dir)
from api.dlpc343x_xpr4 import *
from api.dlpc343x_xpr4_evm import *
from linuxi2c import *
import i2c

class Set(Enum):
    Disabled = 0
    Enabled = 1

def main():
        '''
        This script cycles through available test patterns on the DLPDLCR230NPEVM.
        Raspberry Pi automatically configures each pattern via I2C.
        System will await user input before cycling through each test pattern.
        Test pattern generator (TPG) allows for modification to specific test patterns by modifying command arguments.
        Please review the DLPC3436 Programmer's Guide for more information.
        '''

        gpio_init_enable = False          # Set to FALSE to disable default initialization of Raspberry Pi GPIO pinouts. TRUE by default.
        i2c_time_delay_enable = False    # Set to FALSE to prevent I2C commands from waiting. May lead to I2C bus hangups with some commands if FALSE.
        i2c_time_delay = 0.8             # Lowering this value will speed up I2C commands. Too small delay may lead to I2C bus hangups with some commands.
        protocoldata = ProtocolData()

        def WriteCommand(writebytes, protocoldata):
            '''
            Issues a command over the software I2C bus to the DLPDLCR230NP EVM.
            Set to write to Bus 7 by default
            Some commands, such as Source Select (splash mode) may perform asynchronous access to the EVM's onboard flash memory.
            If such commands are used, it is recommended to provide appropriate command delay to prevent I2C bus hangups.
            '''
            # print ("Write Command writebytes ", [hex(x) for x in writebytes])
            if(i2c_time_delay_enable): 
                time.sleep(i2c_time_delay)
            i2c.write(writebytes)       
            return

        def ReadCommand(readbytecount, writebytes, protocoldata):
            '''
            Issues a read command over the software I2C bus to the DLPDLCR230NP EVM.
            Set to read from Bus 7 by default
            Some commands, such as Source Select (splash mode) may perform asynchronous access to the EVM's onboard flash memory.
            If such commands are used, it is recommended to provide appropriate command delay to prevent I2C bus hangups.
            '''
            # print ("Read Command writebytes ", [hex(x) for x in writebytes])
            if(i2c_time_delay_enable): 
                time.sleep(i2c_time_delay)
            i2c.write(writebytes) 
            readbytes = i2c.read(readbytecount)
            return readbytes

        # ##### ##### Initialization for I2C ##### #####
        # register the Read/Write Command in the Python library
        DLPC343X_XPR4init(ReadCommand, WriteCommand)
        i2c.initialize()
        if(gpio_init_enable): 
            InitGPIO()
        # ##### ##### Command call(s) start here ##### #####  

        print("Cycling DLPC3436 Test Patterns...")
        Summary = WriteDisplayImageCurtain(1,Color.Black)
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Black,   FpgaTestPattern.SolidField,  0)
        Summary = WriteSourceSelect(Source.FpgaTestPattern, Set.Disabled)
        Summary = WriteInputImageSize(1920, 1080)
        time.sleep(1)
        Summary = WriteDisplayImageCurtain(0,Color.Black)
        
        print("Displaying: Solid Field Black")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Black,   FpgaTestPattern.SolidField,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Solid Field White")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.SolidField,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Solid Field Red")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Red,   FpgaTestPattern.SolidField,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Solid Field Green")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Green,   FpgaTestPattern.SolidField,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Solid Field Blue")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Blue,   FpgaTestPattern.SolidField,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Color Bars")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.Black,   FpgaTestPattern.ColorBars,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Horizontal Ramp White 0-255")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.HorizontalRamp,  255)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Vertical Ramp White 0-255")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.VerticalRamp,  255)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Checkerboard White Size 10")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.Checkerboard,  10)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Horizontal Lines")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.HorizontalLines,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Vertical Lines")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.VerticalLines,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Diagonal Lines")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.DiagonalLines,  0)
        input("Press ENTER to continue to next pattern")

        print("Displaying: Actuator Test Pattern")
        Summary = WriteFpgaTestPatternSelect(Set.Disabled,  FpgaTestPatternColor.White,   FpgaTestPattern.ActuatorCalibrationPattern,  0)
        input("Press ENTER to continue to next pattern")

        print("Restoring Raspberry Pi Video Display")
        Summary = WriteDisplayImageCurtain(1,Color.Black)
        Summary = WriteSourceSelect(Source.ExternalParallelPort, Set.Disabled)
        Summary = WriteInputImageSize(1920, 1080)
        Summary = WriteExternalVideoSourceFormatSelect(ExternalVideoFormat.Rgb666)
        time.sleep(1)
        Summary = WriteDisplayImageCurtain(0,Color.Black)
        # ##### ##### Command call(s) end here ##### #####
        i2c.terminate()


if __name__ == "__main__" : main()


