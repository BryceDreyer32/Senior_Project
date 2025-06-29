# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
from PyQt5 import QtWidgets, uic
from PyQt5 import uic
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import sys, os, time
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
sys.path.append(os.path.realpath('python/src/gui'))
import Constants
import FpgaCommunication
import PS4_CtrlGui
import CPU_Usage

class Gui(QtWidgets.QDialog):
    rot1 = False
    rot2 = False
    rot3 = False
    rot4 = False
    drv1 = False
    drv2 = False
    drv3 = False
    drv4 = False
    ps4Connected = False   

    DRV_POWER = 10
    ROT_POWER = 6

    def __init__(self, fpga):
        super().__init__()
        self.fpga = fpga

        print("cwd = " + str(os.getcwd()))

        # Load the UI Page - added path too
        ui_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)

        # Create a MplCanvas instance (matplotlib plot)
        self.canvas = CPU_Usage.CPU_Usage()

        # Add the canvas to the layout in the .ui file
        self.layout = QVBoxLayout(self.cpu_usage)  # Replace 'plot_widget' with your QWidget's name
        self.layout.addWidget(self.canvas)

        self.PS4_Dialog_box.setTextColor(QColor(255, 0, 0))
        self.PS4_Dialog_box.append("Waiting for PS4 to connect to ESP32")

        self.Rotation_Button_1.clicked.connect(self.toggleRotation1)
        self.Rotation_Button_2.clicked.connect(self.toggleRotation2)
        self.Rotation_Button_3.clicked.connect(self.toggleRotation3)
        self.Rotation_Button_4.clicked.connect(self.toggleRotation4)

        self.Drive_Button_1.clicked.connect(self.toggleDrive1)
        self.Drive_Button_2.clicked.connect(self.toggleDrive2)
        self.Drive_Button_3.clicked.connect(self.toggleDrive3)
        self.Drive_Button_4.clicked.connect(self.toggleDrive4)

        self.Arm_Demo_Button.clicked.connect(self.ArmDemoRun)
        self.Arm_Home_Button.clicked.connect(self.ArmGoHome)
        self.Base_Pivot_Slider_1.valueChanged.connect(self.on_baseSlider_change)
        self.Arm_Pivot_Slider.valueChanged.connect(self.on_centerSlider_change)
        self.Wrist_Slider.valueChanged.connect(self.on_wristSlider_change)
        self.Clamp_Slider.valueChanged.connect(self.on_grabberSlider_change)

        self.Estop_Button.clicked.connect(self.eStop)

        self.ps4_thread = PS4_CtrlGui.PS4_CtrlGui()
        self.ps4_thread.ps4_data.connect(self.processPs4Data)
        self.ps4_thread.start()

####################################################################################
# PS4 ROUTINES
####################################################################################
    def setPS4Connected(self):
        # Print connected in green, then change color to redish
        self.PS4_Dialog_box.setTextColor(QColor(0, 255, 0))
        self.PS4_Dialog_box.append("PS4 connected to ESP32")
        self.PS4_Dialog_box.setTextColor(QColor(200, 100, 100))

#        self.LL.setStyleSheet("QProgressBar::chunk "
#                  "{"
#                    "background-color: red;"
#                    "background-color: qconicalgradient(cx:0.5, cy:0.5, angle:0, stop:0 rgba(35, 40, 3, 255), stop:0.16 rgba(136, 106, 22, 255), stop:0.225 rgba(166, 140, 41, 255), stop:0.285 rgba(204, 181, 74, 255), stop:0.345 rgba(235, 219, 102, 255), stop:0.415 rgba(245, 236, 112, 255), stop:0.52 rgba(209, 190, 76, 255), stop:0.57 rgba(187, 156, 51, 255), stop:0.635 rgba(168, 142, 42, 255), stop:0.695 rgba(202, 174, 68, 255), stop:0.75 rgba(218, 202, 86, 255), stop:0.815 rgba(208, 187, 73, 255), stop:0.88 rgba(187, 156, 51, 255), stop:0.935 rgba(137, 108, 26, 255), stop:1 rgba(35, 40, 3, 255));"
#                  "}")

  
    def processPs4Data(self, data):
        data = [int(x) for x in data.split(',')]
        match ( data[0] ):
            case Constants.Constants.PS4_ALL_DATA_FRAME       : self.getPs4Data( data )
            case Constants.Constants.PS4_CONNECT_DATA_FRAME   : self.getPs4ConnectionStatus( data )
            case _                                  : print("Invalid header " + str(data[0]))

    def getPs4Data(self, data):
        if(self.ps4Connected):
            print("[PS4_Ctrl.getPs4Data] Received data:  " + 
                str(data[1]-128).rjust(4) + ", " + 
                str(data[2]-128).rjust(4) + ", " + 
                str(data[3]-128).rjust(4) + ", " + 
                str(data[4]-128).rjust(4) + ", " + 
                str(data[5]).rjust(4))
            self.displayPs4Data(data)
        
    def getPs4ConnectionStatus(self, data):
        if(not self.ps4Connected):
            self.ps4Connected = True
            if( data[1] == 1):
                self.ps4Connected = True
            else:
                self.ps4Connected = False

            self.ps4Battery = data[2]
            self.setPS4Connected()

    def displayPs4Data(self, data):
        # Left stick left-right
        LLval =  256-data[1] if (data[1] < 250 and data[1] > 128) else 0
        LRval =      data[1] if (data[1] > 5   and data[1] < 128) else 0

        self.LL.setValue(LLval)
        self.LR.setValue(LRval)

        # Left stick up-down
        LUval =     data[2] if (data[2] > 5   and data[2] < 128) else 0
        LDval = 256-data[2] if (data[2] < 250 and data[1] > 128) else 0

        self.LU.setValue(LUval)
        self.LD.setValue(LDval)

        # Right stick left-right
        RLval =  256-data[3] if (data[3] < 250 and data[3] > 128) else 0
        RRval =      data[3] if (data[3] > 5   and data[3] < 128) else 0

        self.RL.setValue(RLval)
        self.RR.setValue(RRval)

        # Right stick up-down
        RUval =     data[4] if (data[4] > 5   and data[4] < 128) else 0
        RDval = 256-data[4] if (data[4] < 250 and data[4] > 128) else 0

        self.RU.setValue(RUval)
        self.RD.setValue(RDval)

        # Buttons
        square      = 1 if data[5] & 0x1 else 0
        x           = 1 if data[5] & 0x2 else 0
        circle      = 1 if data[5] & 0x4 else 0
        triangle    = 1 if data[5] & 0x8 else 0

        DarkButton = QPixmap("GUIs/PyQt5/Pictures/DarkGoldButton.png")
        LightButton = QPixmap("GUIs/PyQt5/Pictures/GoldButton.png")
        
        if(square):
            self.Button_Square.setPixmap(DarkButton)
        else:
            self.Button_Square.setPixmap(LightButton)

        if(circle):
            self.Button_Circle.setPixmap(DarkButton)
        else:
            self.Button_Circle.setPixmap(LightButton)

        if(triangle):
            self.Button_Triangle.setPixmap(DarkButton)
        else:
            self.Button_Triangle.setPixmap(LightButton)

        if(x):
            self.Button_X.setPixmap(DarkButton)
        else:
            self.Button_X.setPixmap(LightButton)

        # Text box
        self.PS4_Dialog_box.append( str(data[1]).rjust(4) + ", " + 
                                    str(data[2]).rjust(4) + ", " + 
                                    str(data[3]).rjust(4) + ", " + 
                                    str(data[4]).rjust(4) + ", " + 
                                    str(data[5]).rjust(4) )


####################################################################################
# ROTATION ROUTINES
####################################################################################
    def toggleRotation1(self):
        if(self.rot1):
            self.Rotation_Button_1.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_1.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.ROT_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, powerValue)
            print("Wrote " + hex(self.fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))
        self.rot1 = not self.rot1

    def toggleRotation2(self):
        if(self.rot2):
            self.Rotation_Button_2.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_2.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.ROT_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, powerValue)
        self.rot2 = not self.rot2

    def toggleRotation3(self):
        if(self.rot3):
            self.Rotation_Button_3.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_3.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.ROT_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, powerValue)
        self.rot3 = not self.rot3

    def toggleRotation4(self):
        if(self.rot4):
            self.Rotation_Button_4.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_4.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.ROT_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, powerValue)
        self.rot4 = not self.rot4


####################################################################################
# DRIVE ROUTINES
####################################################################################
    def toggleDrive1(self):
        if(self.drv1):
            self.Drive_Button_1.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_1.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.DRV_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, powerValue)
        self.drv1 = not self.drv1

    def toggleDrive2(self):
        if(self.drv2):
            self.Drive_Button_2.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_2.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.DRV_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, powerValue)
        self.drv2 = not self.drv2

    def toggleDrive3(self):
        if(self.drv3):
            self.Drive_Button_3.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_3.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.DRV_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, powerValue)
        self.drv3 = not self.drv3

    def toggleDrive4(self):
        if(self.drv4):
            self.Drive_Button_4.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_4.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = self.DRV_POWER | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, powerValue)
        self.drv4 = not self.drv4

####################################################################################
# ARM ROUTINES
####################################################################################

    def sliderValueToPWM(self, value, addr):
        # The clock period of the servo pwm is 2^19 (the counter in the FPGA) * the period of the 27MHz FPGA clock
        SERVO_PWM_PERIOD = (2**19) * (1/27e6)

        # The range of servo values is 0.5 - 2.5ms over 20ms. We scale these here since we aren't exactly 20ms cycle
        MIN_SERVO_COUNT = (0.5e-3 / SERVO_PWM_PERIOD) * 255
        MAX_SERVO_COUNT = (2.5e-3 / SERVO_PWM_PERIOD) * 255
        DELTA_SERVO_COUNT = MAX_SERVO_COUNT - MIN_SERVO_COUNT
        match(addr):
            case Constants.Constants.BASE_SERVO_CONTROL_ADDR:
                scale1 = ((Constants.Constants.BASE_SERVO_MAX - Constants.Constants.BASE_SERVO_MIN)/100) * value + Constants.Constants.BASE_SERVO_MIN
                scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
                return int(scale2)
            case Constants.Constants.CENTER_SERVO_CONTROL_ADDR:
                scale1 = ((Constants.Constants.CENTER_SERVO_MAX - Constants.Constants.CENTER_SERVO_MIN)/100) * value + Constants.Constants.CENTER_SERVO_MIN
                scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
                return int(scale2)
            case Constants.Constants.WRIST_SERVO_CONTROL_ADDR:
                scale1 = ((Constants.Constants.WRIST_SERVO_MAX - Constants.Constants.WRIST_SERVO_MIN)/100) * value + Constants.Constants.WRIST_SERVO_MIN
                scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
                return int(scale2)
            case Constants.Constants.GRABBER_SERVO_CONTROL_ADDR:
                scale1 = ((Constants.Constants.GRABBER_SERVO_MAX - Constants.Constants.GRABBER_SERVO_MIN)/100) * value + Constants.Constants.GRABBER_SERVO_MIN
                scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
                return int(scale2)
            case _:
                return 0  
              
    def ArmDemoRun(self):
        print("Go up, nod, and bite demo")
        print("Enabling all...")
        self.fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0xF)

        print("Going up")
        self.fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, self.sliderValueToPWM(20, Constants.Constants.BASE_SERVO_CONTROL_ADDR))
        self.fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(20, Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

        time.sleep(1)

        print("Nod")
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(50, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(100, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(50, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(100, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))

        time.sleep(0.5)

        print("Bite")
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(90, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(0, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(90, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
        time.sleep(0.4)
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(0, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
        time.sleep(0.4)

        print("Going down")
        self.fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, self.sliderValueToPWM(0, Constants.Constants.BASE_SERVO_CONTROL_ADDR))
        self.fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(0, Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

        time.sleep(1)

        print("Disabling all...")
        self.fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0x0)
    
    def ArmGoHome(self):
        print("Arm go-home clicked!")
        self.fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, 20)
        self.fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, 75)
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, 80)
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, 80)

    def on_baseSlider_change(self, value):
        print("Base slider value changed to: " + str(value))
        value = self.sliderValueToPWM(value, Constants.Constants.BASE_SERVO_CONTROL_ADDR)
        print("Writing value " + str(value) + " to BASE_SERVO_CONTROL_ADDR")
        self.fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, value)
        print("Read-back of BASE_SERVO_CONTROL_ADDR = " + str(self.fpga.fpgaRead(Constants.Constants.BASE_SERVO_CONTROL_ADDR)))

    def on_centerSlider_change(self, value):
        print("Center slider value changed to: " + str(value))
        #fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, sliderValueToPWM(event.widget.get(), Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

        value = self.sliderValueToPWM(value, Constants.Constants.CENTER_SERVO_CONTROL_ADDR)
        print("Writing value " + str(value) + " to CENTER_SERVO_CONTROL_ADDR")
        self.fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, value)
        print("Read-back of CENTER_SERVO_CONTROL_ADDR = " + str(self.fpga.fpgaRead(Constants.Constants.CENTER_SERVO_CONTROL_ADDR)))


    def on_wristSlider_change(self, value):
        print("Wrist slider value changed to: " + str(value))
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(value, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))

    def on_grabberSlider_change(self, value):
        print("Grabber slider value changed to: " + str(value))
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(value, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))


####################################################################################
# MISC ROUTINES
####################################################################################
    def eStop(self):
        self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, 0x0)
        self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, 0x0)
        self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, 0x0)
        self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, 0x0)

        self.fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, 0x80)
        self.fpga.fpgaWrite(Constants.Constants.ROTATION1_PWM_TEST_ADDR, 0x80)
        self.fpga.fpgaWrite(Constants.Constants.ROTATION2_PWM_TEST_ADDR, 0x80)
        self.fpga.fpgaWrite(Constants.Constants.ROTATION3_PWM_TEST_ADDR, 0x80)
