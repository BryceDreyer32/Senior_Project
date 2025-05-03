# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
from PyQt5 import QtWidgets, uic
from PyQt5 import uic
from PyQt5.QtGui import QColor
import sys, os, time
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

class Gui(QtWidgets.QDialog):
    rot1 = False
    rot2 = False
    rot3 = False
    rot4 = False
    drv1 = False
    drv2 = False
    drv3 = False
    drv4 = False   

    def __init__(self, fpga):
        super().__init__()
        self.fpga = fpga

        print("cwd = " + str(os.getcwd()))

        # Load the UI Page - added path too
        ui_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)

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


####################################################################################
# PS4 ROUTINES
####################################################################################
    def setPS4Connected(self):
        # Print connected in green, then change color to redish
        self.PS4_Dialog_box.setTextColor(QColor(0, 255, 0))
        self.PS4_Dialog_box.append("PS4 connected to ESP32")
        self.PS4_Dialog_box.setTextColor(QColor(250, 100, 100))
  
    def processPs4Data(self, data):
        # Left stick left-right
        LLval = data[1]-128 if data[1] < 128 else 0
        LRval = data[1]-128 if data[1] >= 128 else 0

        self.LL.tracking = True
        self.LL.value = LLval
        self.LL.sliderPosition = LLval
        self.LL.update()
        self.LL.repaint()                

        self.LR.tracking = True
        self.LR.value = LRval
        self.LR.sliderPosition = LRval
        self.LR.update()
        self.LR.repaint()                

        # Left stick up-down
        LUval = data[2]-128 if data[2] < 128 else 0
        LDval = data[2]-128 if data[2] >= 128 else 0

        self.LU.tracking = True
        self.LU.value = LUval
        self.LU.sliderPosition = LUval
        self.LU.update()
        self.LU.repaint()                

        self.LD.tracking = True
        self.LD.value = LDval
        self.LD.sliderPosition = LDval
        self.LD.update()
        self.LD.repaint() 

        # Right stick left-right
        RLval = data[3]-128 if data[3] < 128 else 0
        RRval = data[3]-128 if data[3] >= 128 else 0

        self.RL.tracking = True
        self.RL.value = RLval
        self.RL.sliderPosition = RLval
        self.RL.update()
        self.RL.repaint()                

        self.RR.tracking = True
        self.RR.value = RRval
        self.RR.sliderPosition = RRval
        self.RR.update()
        self.RR.repaint()                

        # Right stick up-down
        RUval = data[4]-128 if data[4] < 128 else 0
        RDval = data[4]-128 if data[4] >= 128 else 0

        self.RU.tracking = True
        self.RU.value = RUval
        self.RU.sliderPosition = RUval
        self.RU.update()
        self.RU.repaint()                

        self.RD.tracking = True
        self.RD.value = RDval
        self.RD.sliderPosition = RDval
        self.RD.update()
        self.RD.repaint() 

        # Buttons
        square = 1 if data[5] & 0x1 else 0
        circle = 1 if data[5] & 0x2 else 0
        triangle = 1 if data[5] & 0x4 else 0
        x = 1 if data[5] & 0x8 else 0

        DarkButton = "GUIs/PyQt5/Pictures/DarkGoldButton.png"
        LightButton = "GUIs/PyQt5/Pictures/GoldButton.png"
        
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
        self.PS4_Dialog_box.append( str(data[1]-128).rjust(4) + ", " + 
                                    str(data[2]-128).rjust(4) + ", " + 
                                    str(data[3]-128).rjust(4) + ", " + 
                                    str(data[4]-128).rjust(4) + ", " + 
                                    str(data[5]).rjust(4) )


####################################################################################
# ROTATION ROUTINES
####################################################################################
    def toggleRotation1(self):
        if(self.rot1):
            self.Rotation_Button_1.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, 0x0)
        else:
            self.Rotation_Button_1.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, powerValue)
            print("Wrote " + hex(self.fpga.fpgaRead(Constants.Constants.ROTATION0_PWM_TEST_ADDR)))
        self.rot1 = not self.rot1

    def toggleRotation2(self):
        if(self.rot2):
            self.Rotation_Button_2.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_PWM_TEST_ADDR, 0x0)
        else:
            self.Rotation_Button_2.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_PWM_TEST_ADDR, powerValue)
        self.rot2 = not self.rot2

    def toggleRotation3(self):
        if(self.rot3):
            self.Rotation_Button_3.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_PWM_TEST_ADDR, 0x0)
        else:
            self.Rotation_Button_3.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_PWM_TEST_ADDR, powerValue)
        self.rot3 = not self.rot3

    def toggleRotation4(self):
        if(self.rot4):
            self.Rotation_Button_4.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_PWM_TEST_ADDR, 0x0)
        else:
            self.Rotation_Button_4.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_PWM_TEST_ADDR, powerValue)
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
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, powerValue)
        self.drv1 = not self.drv1

    def toggleDrive2(self):
        if(self.drv2):
            self.Drive_Button_2.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_2.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, powerValue)
        self.drv2 = not self.drv2

    def toggleDrive3(self):
        if(self.drv3):
            self.Drive_Button_3.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_3.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, powerValue)
        self.drv3 = not self.drv3

    def toggleDrive4(self):
        if(self.drv4):
            self.Drive_Button_4.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, 0x0)
        else:
            self.Drive_Button_4.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, powerValue)
        self.drv4 = not self.drv4

####################################################################################
# ARM ROUTINES
####################################################################################

    def sliderValueToPWM(value, addr):
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

    def on_baseSlider_change(self, event):
        print("Base slider value changed to: " + str(event.widget.get()))
        value = self.sliderValueToPWM(event.widget.get(), Constants.Constants.BASE_SERVO_CONTROL_ADDR)
        print("Writing value " + str(value) + " to BASE_SERVO_CONTROL_ADDR")
        self.fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, value)
        print("Read-back of BASE_SERVO_CONTROL_ADDR = " + str(self.fpga.fpgaRead(Constants.Constants.BASE_SERVO_CONTROL_ADDR)))

    def on_centerSlider_change(self, event):
        print("Center slider value changed to: " + str(event.widget.get()))
        #fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, sliderValueToPWM(event.widget.get(), Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

        value = self.sliderValueToPWM(event.widget.get(), Constants.Constants.CENTER_SERVO_CONTROL_ADDR)
        print("Writing value " + str(value) + " to CENTER_SERVO_CONTROL_ADDR")
        self.fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, value)
        print("Read-back of CENTER_SERVO_CONTROL_ADDR = " + str(self.fpga.fpgaRead(Constants.Constants.CENTER_SERVO_CONTROL_ADDR)))


    def on_wristSlider_change(self, event):
        print("Wrist slider value changed to: " + str(event.widget.get()))
        self.fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, self.sliderValueToPWM(event.widget.get(), Constants.Constants.WRIST_SERVO_CONTROL_ADDR))

    def on_grabberSlider_change(self, event):
        print("Grabber slider value changed to: " + str(event.widget.get()))
        self.fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, self.sliderValueToPWM(event.widget.get(), Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))


