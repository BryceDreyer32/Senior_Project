import matplotlib.pyplot as plt
import json

# Load saved parameters
with open("motor_params.json") as f:
    params = json.load(f)

J = params["J"]
friction_table = params["friction_table"]

# Correct motor constants
KTORQUE = 0.01042  # Nm/A
PWM_TO_CURRENT = 20.0  # A per PWM unit

# Extract arrays
pos = [p for p, b, tc in friction_table]
b_vals = [b for p, b, tc in friction_table]
tc_vals = [tc for p, b, tc in friction_table]

# Convert if needed
b_vals_Nm = [b * KTORQUE for b in b_vals]
tc_vals_Nm = [tc * KTORQUE for tc in tc_vals]

# Plot viscous friction
plt.figure()
plt.plot(pos, b_vals_Nm, 'o-')
plt.xlabel("Position (rad)")
plt.ylabel("b (Nm·s/rad)")
plt.title("Viscous friction coefficient vs position")
plt.grid(True)

# Plot Coulomb friction
plt.figure()
plt.plot(pos, tc_vals_Nm, 'o-')
plt.xlabel("Position (rad)")
plt.ylabel("τc (Nm)")
plt.title("Coulomb friction vs position")
plt.grid(True)

plt.show()
