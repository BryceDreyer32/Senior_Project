#        uic.loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)
#        self.layout = QVBoxLayout(self.cpu_usage)  # Replace 'plot_widget' with your QWidget's name
import sys
import psutil
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.animation import FuncAnimation
from collections import deque

# Example Matplotlib Plot with Animated CPU Usage for Each Core
class MplCanvas(FigureCanvas):
    def __init__(self):
        # Create a figure and axis for plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        super().__init__(self.fig)
        self.setParent(None)

        self.xdata = deque(maxlen=10)  # Store the last 10 seconds of data (time)
        self.ydata = [deque(maxlen=10) for _ in range(psutil.cpu_count(logical=False))]  # One deque per CPU core

        # Set figure background to dark transparent gray
        self.fig.patch.set_facecolor('#101010')  # Dark gray background
        self.fig.patch.set_alpha(0.5)  # Make it slightly transparent

        # Set axes background to dark gray
        self.ax.set_facecolor('#101010')
        
        # Set up the plot
        self.ax.set_title("Real-time CPU Usage per Core", fontsize=8)  # Adjust title font size
        self.ax.set_xlabel("Time (s)", fontsize=8)  # Adjust x-axis label font size
        self.ax.set_ylabel("CPU Usage (%)", fontsize=8)  # Adjust y-axis label font size
        self.ax.set_ylim(0, 100)
        self.ax.tick_params(axis='x', labelsize=8)
        self.ax.tick_params(axis='y', labelsize=8)

        # Initialize lines for each CPU core
        self.lines = [self.ax.plot([], [], label=f"Core {i+1}")[0] for i in range(psutil.cpu_count(logical=False))]

        # Set the initial x-axis limit (this will now change dynamically)
        self.ax.set_xlim(0, 10)  # Show the last 10 seconds

        self.anim = FuncAnimation(self.fig, self.update, interval=1000, blit=True)

    def update(self, frame):
        # Get current CPU usage per core
        cpu_usage_per_core = psutil.cpu_percent(interval=None, percpu=True)

        # Add new data point to the deque for each core
        if len(self.xdata) == 10:
            self.xdata.append(self.xdata[-1] + 1)  # Increment time for the new data point
        else:
            self.xdata.append(len(self.xdata))  # Add time data for the first few frames

        # Update ydata for each core
        for i, cpu_usage in enumerate(cpu_usage_per_core):
            self.ydata[i].append(cpu_usage)

        # Update the plot data for each core
        for i in range(len(self.lines)):
            self.lines[i].set_data(list(self.xdata), list(self.ydata[i]))  # Convert deque to list to avoid shape issues

        # Adjust the x-axis to "scroll" by shifting the window
        # The window size will remain fixed at 10 seconds
        self.ax.set_xlim(self.xdata[0], self.xdata[0] + 10)  # Set the xlim to the first time value + 10 seconds

        # Return the updated line objects for animation
        return self.lines

class MainWindow(QDialog):
    def __init__(self):
        super().__init__()

        # Load the .ui file created by PyQt Designer
        loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)

        # Create a MplCanvas instance (matplotlib plot)
        self.canvas = MplCanvas()

        # Add the canvas to the layout in the .ui file
        self.layout = QVBoxLayout(self.cpu_usage)  # Replace 'plot_widget' with your QWidget's name
        self.layout.addWidget(self.canvas)

        # Set the window properties
        self.setWindowTitle("Real-Time CPU Usage per Core in PyQt5")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
