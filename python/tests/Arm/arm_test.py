import sys, os
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QDialog, QSlider
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication


class MyWindow(QDialog):
    def __init__(self):
        super().__init__()
        # Load the .ui file dynamically
        uic.loadUi("python/tests/Arm/arm_gui.ui", self)
        
        # Slot connections
        self.baseSlider = self.findChild(QSlider, "baseSlider")
        self.baseSlider.valueChanged.connect(self.on_baseSlider_value_changed)
        self.centerSlider = self.findChild(QSlider, "centerSlider")
        self.centerSlider.valueChanged.connect(self.on_centerSlider_value_changed)
        self.wristSlider = self.findChild(QSlider, "wristSlider")
        self.wristSlider.valueChanged.connect(self.on_wristSlider_value_changed)
        self.grabberSlider = self.findChild(QSlider, "grabberSlider")
        self.grabberSlider.valueChanged.connect(self.on_grabberSlider_value_changed)
        self.extendButton.clicked.connect(self.on_extend_click)
        self.contractButton.clicked.connect(self.on_contract_click)

    def on_baseSlider_value_changed(self, value):
        print("Base slider value changed to: " + str(value))
        fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, value)

    def on_centerSlider_value_changed(self, value):
        print("Center slider value changed to: " + str(value))
        fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, value)

    def on_wristSlider_value_changed(self, value):
        print("Wrist slider value changed to: " + str(value))
        fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, value)

    def on_grabberSlider_value_changed(self, value):
        print("Grabber slider value changed to: " + str(value))
        fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, value)

    def on_extend_click(self):
        print("Extend clicked!")
        fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, 70)
        fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, 25)
        fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, 60)
        fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, 50)

    def on_contract_click(self):
        print("Contract clicked!")
        fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, 20)
        fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, 75)
        fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, 80)
        fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, 80)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())





'''

import os
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication

#print("cwd = " + os.getcwd())

class Arm:
    def __init__(self):



Form, Window = uic.loadUiType("python/tests/Arm/arm_gui.ui")

app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
app.exec()
'''