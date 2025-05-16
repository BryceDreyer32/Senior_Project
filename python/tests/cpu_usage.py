import psutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque

# Configuration
interval_sec = 1
window_size_sec = 10  # Show last 10 seconds
cpu_count = psutil.cpu_count(logical=True)

# Initialize data containers using deque with maxlen
timestamps = deque(maxlen=window_size_sec)
cpu_usage_history = [deque(maxlen=window_size_sec) for _ in range(cpu_count)]

# Setup plot
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f'CPU {i}')[0] for i in range(cpu_count)]

ax.set_ylim(0, 100)
ax.set_xlim(0, window_size_sec)
ax.set_ylabel('CPU Usage (%)')
ax.set_xlabel('Time (s)')
ax.set_title('Per-CPU Usage (Last 10 Seconds)')
ax.legend(loc='upper right')

start_time = time.time()

def update(frame):
    current_time = time.time() - start_time
    timestamps.append(current_time)

    # Get per-CPU usage
    usage = psutil.cpu_percent(percpu=True)
    for i, val in enumerate(usage):
        cpu_usage_history[i].append(val)
        lines[i].set_data(timestamps, cpu_usage_history[i])

    # Set x-axis to last 10 seconds
    if len(timestamps) > 1:
        ax.set_xlim(timestamps[0], timestamps[-1])
    
    return lines

# Animation
ani = animation.FuncAnimation(fig, update, interval=interval_sec * 1000, blit=False)

plt.tight_layout()
plt.show()
