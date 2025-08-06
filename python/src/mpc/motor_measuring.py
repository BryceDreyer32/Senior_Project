import time 
import numpy as np 
import math, sys, os, json
from sklearn.linear_model import LinearRegression 
from scipy.signal import savgol_filter
sys.path.append(os.path.realpath('python/src'))
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication
from hal.hal import HAL

# === USER PARAMETERS === 
PWM_MIN_MOVE = 4          # barely moves 
PWM_MAX_SAFE = 12         # danger limit 
PWM_ACCEL_TEST = 10       # safe acceleration test value 
PWM_HOLD_MAX = 8          # safe for friction sweep 
TEST_DURATION_ACCEL = 0.5 # seconds 
ENCODER_RESOLUTION = 4096 # 12-bit encoder 
GEAR_RATIO = 120/27
Rmotor = 0.1              # Neo 550 has 0.1 Ohm resistance
#Kv = 917 * 2 * math.pi    # Neo 550 Kv is 917 rpm/V - converted to rad/s/V
#Kt = Kv * math.sqrt(Rmotor)
Kt = 1/(917*2*math.pi/60)
TORQUE_CONSTANT = Kt      # Nm per Amp 
PWM_TO_CURRENT = 0.5      # Amps per PWM unit 
OPERATING_RANGE_DEG = 30  # range of motion to test (+/- degrees) 

RUNS_PER_DIRECTION = 2
TRIALS_PER_PWM = 10
SAMPLE_INTERVAL = 0.002
BIN_WIDTH_DEG = 5

MOTOR = 2

class MotorMeasuring:
    def __init__(self, hal):
        self.hal = hal

    def set_pwm(self, motor, pwm_value, direction):
        if(pwm_value == 0):
            self.hal.stop_motor(motor)
        else:
            self.hal.run_motor(motor, pwm_value, direction)
    
    def read_encoder(self, motor): 
        return self.hal.get_angle(motor)
 

    # === Helper === 
    def counts_to_radians(self, counts): 
        return (counts / ENCODER_RESOLUTION) * 2 * np.pi / GEAR_RATIO 

    def smooth_and_diff(self, theta_arr, t_arr):
        win_len = min(len(theta_arr)//2*2+1, 21)
        theta_s = savgol_filter(theta_arr, win_len, 3)
        omega = np.gradient(theta_s, t_arr)
        omega_s = savgol_filter(omega, win_len, 3)
        alpha = np.gradient(omega_s, t_arr)
        return omega_s, alpha

    def within_operating_range(self, theta, theta_center): 

        return abs(theta - theta_center) <= np.deg2rad(OPERATING_RANGE_DEG / 2) 


    def measure_inertia(self, motor):
        print("\n--- Measuring Inertia J ---")
        J_estimates = []
        for direction in [1, 0]:
            for _ in range(RUNS_PER_DIRECTION):
                self.set_pwm(motor, 0, 0); time.sleep(0.5)
                theta_list, t_list = [], []
                start = time.time()
                self.set_pwm(motor, PWM_ACCEL_TEST, direction)
                while time.time() - start < TEST_DURATION_ACCEL:
                    t_list.append(time.time() - start)
                    theta_list.append(self.counts_to_radians(self.read_encoder(motor)))
                    time.sleep(SAMPLE_INTERVAL)
                self.set_pwm(motor, 0, direction)

                theta_arr = np.array(theta_list)
                t_arr = np.array(t_list)
                omega, alpha = self.smooth_and_diff(theta_arr, t_arr)

                tau_applied = direction * PWM_ACCEL_TEST * PWM_TO_CURRENT * TORQUE_CONSTANT
                # Very rough friction guess
                tau_friction = 0.02*np.sign(omega) + 0.01*omega
                tau_net = tau_applied - tau_friction

                mask = np.abs(alpha) > 1e-3
                if np.sum(mask) < 5:
                    continue
                reg = LinearRegression().fit(alpha[mask].reshape(-1,1), tau_net[mask])
                if reg.coef_[0] > 0:
                    J_estimates.append(reg.coef_[0])

        if not J_estimates:
            raise RuntimeError("No valid J estimates — try higher PWM_ACCEL_TEST or check encoder.")
        J_final = np.median(J_estimates)
        print(f"All J estimates: {[f'{j:.3e}' for j in J_estimates]}")
        print(f"Median J = {J_final:.6e} kg·m²")
        return J_final


    def measure_friction_vs_position(self, motor):
        print("\n--- Measuring Friction vs Position ---")
        theta_center = self.counts_to_radians(self.read_encoder(motor))
        data = []

        pwm_levels = np.linspace(PWM_MIN_MOVE + 0.5, PWM_HOLD_MAX, 4)

        for pwm_test in range(PWM_MIN_MOVE+1, PWM_HOLD_MAX):
            for direction in [1, 0]:
                for trial in range(TRIALS_PER_PWM):
                    self.set_pwm(motor, pwm_test, direction)
                    time.sleep(0.3)

                    while True:
                        theta = self.counts_to_radians(self.read_encoder(motor))
                        if abs(theta - theta_center) > np.deg2rad(OPERATING_RANGE_DEG / 2):
                            break

                        t0 = time.time()
                        theta0 = theta
                        time.sleep(0.02)
                        theta1 = self.counts_to_radians(self.read_encoder(motor))
                        omega = (theta1 - theta0) / 0.02

                        current = pwm_test * PWM_TO_CURRENT
                        torque = current * TORQUE_CONSTANT
                        data.append((theta, abs(omega), abs(torque)))

                    self.set_pwm(motor, 0, direction)
                    time.sleep(0.3)  # rest between sweeps

        # Convert to arrays
        data = np.array(data)
        positions, speeds, torques = data[:, 0], data[:, 1], data[:, 2]

        # Position binning
        bin_w = np.deg2rad(BIN_WIDTH_DEG)
        bins = np.arange(positions.min(), positions.max() + bin_w, bin_w)

        friction_table = []
        for pb in bins:
            mask = (positions >= pb) & (positions < pb + bin_w)
            if np.sum(mask) >= 5:
                abs_omega = speeds[mask].reshape(-1, 1)
                abs_torque = torques[mask]
                reg = LinearRegression().fit(abs_omega, abs_torque)
                friction_table.append((pb + bin_w / 2, reg.coef_[0], reg.intercept_))

        print("\nFriction table:")
        for pos, b, tc in friction_table:
            print(f"pos={pos:.4f} rad, b={b:.6f}, tau_c={tc:.6f}")

        return friction_table


#    def measure_friction_vs_position(self, motor):
#        print("\n--- Measuring Friction vs Position ---")
#        theta_center = self.counts_to_radians(self.read_encoder(motor))
#        data = []
#
#        for pwm_test in range(PWM_MIN_MOVE+1, PWM_HOLD_MAX):
#            for direction in [1, 0]:
#                self.set_pwm(motor, pwm_test, direction)
#                time.sleep(0.3)
#                while True:
#                    theta = self.counts_to_radians(self.read_encoder(motor))
#                    if abs(theta - theta_center) > np.deg2rad(OPERATING_RANGE_DEG/2):
#                        break
#                    t0 = time.time()
#                    theta0 = theta
#                    time.sleep(0.02)
#                    theta1 = self.counts_to_radians(self.read_encoder(motor))
#                    omega = (theta1 - theta0) / 0.02
#                    current = pwm_test * PWM_TO_CURRENT
#                    torque = current * TORQUE_CONSTANT
#                    data.append((theta, abs(omega), abs(torque)))
#                self.set_pwm(motor, 0, 1); time.sleep(0.3)
#
#        data = np.array(data)
#        positions, speeds, torques = data[:,0], data[:,1], data[:,2]
#        bin_w = np.deg2rad(BIN_WIDTH_DEG)
#        bins = np.arange(positions.min(), positions.max()+bin_w, bin_w)
#
#        friction_table = []
#        for pb in bins:
#            mask = (positions >= pb) & (positions < pb+bin_w)
#            if np.sum(mask) >= 5:
#                abs_omega = speeds[mask].reshape(-1,1)
#                abs_torque = torques[mask]
#                reg = LinearRegression().fit(abs_omega, abs_torque)
#                friction_table.append((pb+bin_w/2, reg.coef_[0], reg.intercept_))
#
#        print("\nFriction table:")
#        for pos,b,tc in friction_table:
#            print(f"pos={pos:.4f} rad, b={b:.6f}, tau_c={tc:.6f}")
#        return friction_table


#    # === Measure Inertia === 
#    def measure_inertia(self, motor): 
#        print("\nMeasuring inertia J (with load attached)...") 
#        self.set_pwm(motor, 0, 1) 
#        time.sleep(1.0) 
#
#        theta_list, time_list = [], [] 
#        start_time = time.time() 
#    
#        self.set_pwm(motor, PWM_ACCEL_TEST, 1) 
#        while time.time() - start_time < TEST_DURATION_ACCEL: 
#            now = time.time() - start_time 
#            theta_list.append(self.counts_to_radians(self.read_encoder(motor))) 
#            time_list.append(now) 
#            time.sleep(0.001) 
#
#        self.set_pwm(motor, 0, 1) 
#
#        # Following implements a Savitzky-Golay filter (to filter out noise on angle data)
#        theta_arr = np.array(theta_list) 
#        t_arr = np.array(time_list) 
#        theta_smoothed = savgol_filter(theta_arr, window_length=21, polyorder=3)
#        omega = np.gradient(theta_smoothed, t_arr)
#        omega_smoothed = savgol_filter(omega, window_length=21, polyorder=3)
#        alpha = np.gradient(omega_smoothed, t_arr)
#  
#        current = PWM_ACCEL_TEST * PWM_TO_CURRENT 
#        torque = current * TORQUE_CONSTANT 
#
#        alpha_mean = np.mean(alpha[5:]) 
#        if abs(alpha_mean) < 1e-6: 
#            raise RuntimeError("Acceleration too small — try higher PWM_ACCEL_TEST") 
#
#        J_est = torque / alpha_mean 
#        print(f"Estimated J = {J_est:.6f} kg·m²") 
#        return J_est 
#
#    # === Measure Friction vs Position === 
#    def measure_friction_vs_position(self, motor, J):
#        print("\nMeasuring viscous and Coulomb friction as function of position...")
#
#        theta_center = self.counts_to_radians(self.read_encoder(motor))
#        friction_data = []
#
#        for pwm_test in range(PWM_MIN_MOVE+1, PWM_HOLD_MAX):
#            for direction in [1, 0]:
#                self.set_pwm(motor, pwm_test, direction)
#                time.sleep(0.3)  # start moving
#
#                t_start = time.time()
#                while True:
#                    theta = self.counts_to_radians(self.read_encoder(motor))
#                    if not self.within_operating_range(theta, theta_center):
#                        break
#                    # Instantaneous velocity
#                    t0 = time.time()
#                    theta0 = theta
#                    time.sleep(0.02)
#                    theta1 = self.counts_to_radians(self.read_encoder(motor))
#                    omega = (theta1 - theta0) / 0.02
#
#                    current = pwm_test * PWM_TO_CURRENT
#                    torque = current * TORQUE_CONSTANT
#
#                    friction_data.append((theta, abs(omega), abs(torque)))
#
#                    if time.time() - t_start > 2.0:
#                        break  # avoid overheating
#
#                self.set_pwm(motor, 0, 1) 
#                time.sleep(0.3)
#
#        # Convert to arrays
#        friction_data = np.array(friction_data)
#        positions = friction_data[:, 0]
#        speeds = friction_data[:, 1]
#        torques = friction_data[:, 2]
#
#        # Position binning
#        bin_width_rad = np.deg2rad(6)  # ~6 degrees wide
#        pos_bins = np.arange(positions.min(), positions.max() + bin_width_rad, bin_width_rad)
#
#        friction_table = []
#        for pb in pos_bins:
#            mask = (positions >= pb) & (positions < pb + bin_width_rad)
#            if np.sum(mask) >= 5:  # enough samples
#                abs_omega = speeds[mask].reshape(-1, 1)
#                abs_torque = torques[mask]
#                reg = LinearRegression().fit(abs_omega, abs_torque)
#                b_est = reg.coef_[0]
#                tau_c_est = reg.intercept_
#                friction_table.append((pb + bin_width_rad/2, b_est, tau_c_est))
#
#        print("\nFriction table (pos rad, b, tau_c):")
#        for pos, b_val, tc_val in friction_table:
#            print(f"{pos:.4f}, {b_val:.6f}, {tc_val:.6f}")
#
#        return friction_table
 

# === Main === 
if __name__ == "__main__": 
    fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
    hal = HAL(fpga)
 
    MOTOR = 2
    motor_meas = MotorMeasuring(hal)
    motor_meas.set_pwm(MOTOR, 0, 0)
    time.sleep(5)

#    J = motor_meas.measure_inertia(MOTOR) 
    J = 1.309405e-06
    friction_table = motor_meas.measure_friction_vs_position(MOTOR) 
    params = {"J": float(J), "friction_table": friction_table}
    with open("motor_params.json","w") as f:
        json.dump(params, f, indent=2)
    print("\nSaved motor_params.json")

    print("\nFinal estimated parameters for MPC:") 
    print(f"J = {J:.6e} kg·m²") 
    print("Friction table:", friction_table) 

 