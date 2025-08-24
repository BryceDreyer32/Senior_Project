"""
Orange Pi BLDC Identification Script (Spark MAX + NEO 550)
----------------------------------------------------------
This script is designed to experimentally identify the BLDC system parameters (J, b, KT) 
for use in an MPC controller. It assumes:
  - Motor: Rev NEO 550, KV = 917 rpm/V, Rs = 0.1 Ohm
  - Driver: Spark MAX, controlled via PWM (servo style)
  - Controller: Orange Pi
  - Gear ratio: 120:27
  - Load encoder: 12-bit absolute encoder

PWM and encoder I/O functions are placeholders — you should connect them
with your existing library.
"""

import time
import csv
import math
import argparse
from pathlib import Path
import numpy as np
import sys, os
sys.path.append(os.path.realpath('python/src'))
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication
import matplotlib.pyplot as plt
from hal.hal import HAL
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# ---------------- CONFIG ----------------
PWM_MIN = 12     # minimum SparkMax PWM command
PWM_MAX = 50     # maximum safe SparkMax PWM command
R = 0.1          # motor resistance (ohm)
KV = 917.0       # motor KV (rpm/V)
V_SUPPLY = 12.0  # supply voltage
ENC_RES = 4096   # 12-bit encoder
GEAR_RATIO = 27.0 / 120.0
DIRECTION_SIGN = -1  # try -1 if signs are inverted
MAX_COUNT_DELTA = 2500  # maximum plausible encoder delta per sample

# Trial shaping (gentle)
HARD_ANGLE_CAP = np.deg2rad(120)  # absolute stop no matter what
SAMPLE_DT = 0.005

# Randomized trial ranges
PWM_MIN_TRIAL = 0.12  # normalized duty cycle (0..1)
PWM_MAX_TRIAL = 0.25
ANGLE_MIN = np.deg2rad(45)
ANGLE_MAX = np.deg2rad(90)
TRIAL_TIME_MIN = 0.25
TRIAL_TIME_MAX = 0.90

# Derived constants
Ke = 60.0 / (2.0 * np.pi * KV)   # V·s/rad
Kt = Ke                          # N·m/A
g_t = GEAR_RATIO

fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
hal = HAL(fpga, None)
MOTOR = 2


# Limits
MAX_COUNT_DELTA = 1200  # ~105° per sample at 4096 CPR; tighten to avoid blow-ups

class ThetaTracker:
    def __init__(self, enc_res=ENC_RES, direction_sign=DIRECTION_SIGN, max_delta=MAX_COUNT_DELTA):
        self.enc_res = int(enc_res)
        self.sign = 1 if direction_sign >= 0 else -1
        self.max_delta = int(max_delta)
        self.count_prev = None
        self.theta = 0.0  # radians cumulative

    def reset(self, count0):
        self.count_prev = int(count0)
        self.theta = 0.0

    def update(self, count_now):
        c = int(count_now)
        if self.count_prev is None:
            self.count_prev = c
            return self.theta
        d = c - self.count_prev
        # unwrap to nearest (handle wrap-around)
        d = (d + self.enc_res//2) % self.enc_res - self.enc_res//2
        # clip per-sample jump
        if d >  self.max_delta: d =  self.max_delta
        if d < -self.max_delta: d = -self.max_delta
        self.count_prev = c
        self.theta += self.sign * (2*np.pi/self.enc_res) * d
        return self.theta

# ---------------- HARDWARE HOOKS ----------------
# Replace these with your own library calls

def pwm_init(nada):
    """Initialize PWM output to Spark MAX."""
    pass

def pwm_set_duty(duty, pwm_channel, dir_pin):
    """
    Set motor duty command.

    Parameters
    ----------
    duty : float
        Normalized duty in [-1.0, 1.0]. 
        -1.0 = full reverse, +1.0 = full forward.
    pwm_channel : int or object
        Handle or channel for your PWM hardware.
    dir_pin : int or object
        Handle or pin for your direction GPIO.
    """

    PWM_MIN = 12   # minimum PWM value that moves motor
    PWM_MAX = 50   # maximum safe PWM value

    # Clip duty to [-1.0, 1.0]
    duty = max(-1.0, min(1.0, duty))

    # Determine direction and magnitude
    if duty >= 0:
        direction = 1   # forward
        mag = duty
    else:
        direction = 0   # reverse
        mag = -duty

    # Scale [0.0 … 1.0] → [PWM_MIN … PWM_MAX]
    pwm_val = int(PWM_MIN + mag * (PWM_MAX - PWM_MIN)) if mag > 0 else 0

    # --- USER HOOKS ---
    # Replace with your own library calls:
    if(pwm_val <= PWM_MIN):
        hal.stop_motor(MOTOR)
    elif(pwm_val >= PWM_MAX):
        hal.run_motor(MOTOR, PWM_MAX, direction)
    else:
        hal.run_motor(MOTOR, pwm_val, direction)

def encoder_init():
    """Initialize the 12-bit encoder."""
    pass

def encoder_read():
    """Read encoder count (0–4095)."""
    return hal.get_angle(MOTOR)


# ===========================
# Helper functions
# ===========================

# -------------------
# Encoder + Safety
# -------------------
ENCODER_MAX = 4096
DEG_PER_COUNT = 360.0 / ENCODER_MAX
MAX_VALID_DELTA = 2500   # reject if encoder jumps more than this between samples

def unwrap_encoder(theta_counts):
    """Unwraps 12-bit encoder counts into continuous angles (rad)."""
    theta_counts = np.asarray(theta_counts, dtype=float)
    diffs = np.diff(theta_counts)
    jumps = np.where(np.abs(diffs) > ENCODER_MAX / 2)[0]

    corrected = theta_counts.copy()
    offset = 0
    for j in jumps:
        if diffs[j] > 0:
            offset -= ENCODER_MAX
        else:
            offset += ENCODER_MAX
        corrected[j+1:] += offset
    return corrected * (2*np.pi / ENCODER_MAX)

def reject_glitches(theta_counts, times):
    """Reject garbage encoder data (large unrealistic jumps)."""
    filtered = [theta_counts[0]]
    filtered_t = [times[0]]

    for i in range(1, len(theta_counts)):
        delta = abs(theta_counts[i] - filtered[-1])
        if delta < MAX_VALID_DELTA:
            filtered.append(theta_counts[i])
            filtered_t.append(times[i])
        # else discard sample
    return np.array(filtered), np.array(filtered_t)

def median_filter(arr, k=3):
    """Simple median filter for encoder noise."""
    arr = np.asarray(arr)
    out = arr.copy()
    half = k // 2
    for i in range(half, len(arr)-half):
        out[i] = np.median(arr[i-half:i+half+1])
    return out


def active_brake(last_duty, pwm_channel, dir_pin, t_brake=0.08):
    # brief reverse to arrest motion; scales with last command, clamped
    mag = min(0.4, max(0.2, abs(last_duty)*0.6))
    pwm_set_duty(-np.sign(last_duty)*mag, pwm_channel, dir_pin)
    time.sleep(t_brake)
    pwm_set_duty(0.0, pwm_channel, dir_pin)

def counts_to_continuous_rad(counts, max_delta=MAX_COUNT_DELTA):
    counts = np.asarray(counts, dtype=np.int64)
    deltas = np.zeros_like(counts)
    for k in range(1, len(counts)):
        d = counts[k]-counts[k-1]
        d = (d + ENC_RES//2) % ENC_RES - ENC_RES//2
        d = np.clip(d, -max_delta, max_delta)
        deltas[k] = deltas[k-1] + d
    return DIRECTION_SIGN*(2*np.pi/ENC_RES)*deltas

def compute_velocity(theta, t, window=5):
    """Compute angular velocity using smoothed differentiation."""
    theta = np.asarray(theta)
    t = np.asarray(t)

    if len(theta) < 2:
        return np.zeros_like(theta)

    if len(theta) <= window:
        omega = np.gradient(theta, t)
    else:
        smoothed = np.convolve(theta, np.ones(window)/window, mode="valid")
        t_mid = t[window//2: -window//2]
        omega = np.gradient(smoothed, t_mid)
        # Pad to match original length
        pad_left = [omega[0]] * (window//2)
        pad_right = [omega[-1]] * (len(theta) - len(smoothed) - len(pad_left))
        omega = np.concatenate([pad_left, omega, pad_right])
    return omega


# ===========================
# Run randomized small-angle trials
# ===========================
# Tune these at top of your file if you like
TRIAL_TIME_MIN = 0.6   # ensure we see the rise
TRIAL_TIME_MAX = 1.5   # safety cap

def run_trials(num_trials=20, pwm_channel=0, dir_pin=0):
    all_data = []
    for k in range(num_trials):
        duty = float(np.random.uniform(PWM_MIN_TRIAL, PWM_MAX_TRIAL))
        duty *= 1 if np.random.rand() > 0.5 else -1
        target = float(np.random.uniform(ANGLE_MIN, ANGLE_MAX))
        print(f"--- Trial {k+1}/{num_trials}: duty={duty:.2f}, target={np.rad2deg(target):.1f}° ---")

        # init logger
        timestamps = []
        counts = []

        # init tracker
        tracker = ThetaTracker()
        c0 = encoder_read()
        tracker.reset(c0)

        # start PWM
        pwm_set_duty(duty, pwm_channel, dir_pin)
        t0 = time.time()

        # seed first sample
        timestamps.append(0.0)
        counts.append(c0)
        theta0 = 0.0  # tracker.theta already zeroed

        reached = False
        while True:
            t_now = time.time() - t0
            if t_now >= TRIAL_TIME_MAX:
                break
            c = encoder_read()
            counts.append(c)
            theta = tracker.update(c)
            timestamps.append(t_now)

            delta = abs(theta - theta0)
            # only allow target stop AFTER we’ve seen enough of the transient
            if (t_now >= TRIAL_TIME_MIN) and (delta >= target):
                reached = True
                break
            # emergency angle cap
            if delta >= HARD_ANGLE_CAP:
                print("  [safety] hard angle cap hit")
                break

            time.sleep(SAMPLE_DT)

        # stop & brake
        pwm_set_duty(0.0, pwm_channel, dir_pin)
        active_brake(duty, pwm_channel, dir_pin)

        # pack trial
        T = np.asarray(timestamps, dtype=np.float64)
        C = np.asarray(counts, dtype=np.int64)
        all_data.append({"timestamps": T, "counts": C, "duty": duty, "hit_target": reached})

        time.sleep(0.4)  # settle
    return all_data



# ===========================
# Nonlinear ω(t) fitting
# ===========================
def omega_model(t, tau, omega_inf):
    return omega_inf * (1.0 - np.exp(-t / tau))


def estimate_trial_velocity(trial):
    t = trial["timestamps"]
    theta = counts_to_continuous_rad(trial["counts"])  # ok to reuse your existing converter
    # discard if too short or barely moved
    if len(t) < 10 or (abs(theta[-1]-theta[0]) < np.deg2rad(20)):
        return None, None, None, None, theta, t, np.zeros_like(t)

    omega, theta_s = compute_velocity(theta, t)

    # robust window: 5–90th percentile of time to avoid ends
    t_lo, t_hi = np.percentile(t, [5, 90])
    m = (t >= t_lo) & (t <= t_hi)
    if m.sum() < 8:
        m = np.ones_like(t, dtype=bool)

    w = omega[m]; tt = t[m]
    if np.allclose(w, 0.0, atol=1e-3):
        return None, None, None, None, theta_s, t, omega

    # percentile clip on ω
    lo, hi = np.percentile(w, [5, 95])
    sel = (w >= lo) & (w <= hi)
    w_fit = w[sel]; t_fit = tt[sel]
    if len(t_fit) < 8:
        return None, None, None, None, theta_s, t, omega

    omega_inf_guess = max(np.percentile(w_fit, 90), 1e-3)
    tau_guess = 0.12

    try:
        popt, _ = curve_fit(
            omega_model, t_fit, w_fit,
            p0=[tau_guess, omega_inf_guess],
            bounds=([8e-3, 1e-4], [1.5, 1e5]),
            maxfev=3000
        )
        tau, omega_inf = float(popt[0]), float(popt[1])
        V_eff = abs(trial["duty"]) * V_SUPPLY
        if V_eff < 1e-3: return None, None, None, None, theta_s, t, omega
        slope = omega_inf / V_eff
        if tau <= 0 or slope <= 0:
            return None, None, None, None, theta_s, t, omega
        return tau, slope, omega_inf, V_eff, theta_s, t, omega
    except Exception:
        return None, None, None, None, theta_s, t, omega

def recover_physical_params(trials):
    taus, slopes = [], []
    for tr in trials:
        tau, slope, *_ = estimate_trial_velocity(tr)
        if (tau is not None) and (slope is not None):
            taus.append(tau); slopes.append(slope)

    if len(taus) < 4:
        raise RuntimeError("Too few valid trials; try more or lower PWM / angle.")

    taus = np.array(taus); slopes = np.array(slopes)
    # trim outliers
    keep_t = (taus >= np.percentile(taus, 10)) & (taus <= np.percentile(taus, 90))
    keep_s = (slopes >= np.percentile(slopes, 10)) & (slopes <= np.percentile(slopes, 90))
    keep = keep_t & keep_s
    taus = taus[keep]; slopes = slopes[keep]

    tau_hat = float(np.median(taus))
    slope_hat = float(np.median(slopes))

    B = g_t * Kt / R
    A = B / max(slope_hat, 1e-9)
    b = A - g_t * Kt * Ke / R
    J = tau_hat * A
    return dict(J=J, b=b, tau=tau_hat, slope=slope_hat)

# ===========================
# Main
# ===========================
if __name__=="__main__":
    pwm_init(0)
    try:
        data = run_trials(num_trials=50)
        results = recover_physical_params(data)
        print("Estimated parameters:")
        print(f"J = {results['J']:.6f}")
        print(f"b = {results['b']:.6f}")
        print(f"tau = {results['tau']:.6f} s")
        print(f"slope = {results['slope']:.6f} rad/s per V")

        # Plot first trial
        tau, slope, omega_inf, V_eff, theta_s, t, omega = estimate_trial_velocity(data[0])
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t, theta_s, label='theta smooth')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(t, omega, label='omega')
        plt.hlines(omega_inf, t[0], t[-1], colors='k', linestyle='--', label='omega_inf')
        plt.legend()
        plt.show()
    finally:
        pwm_set_duty(0, 0, 0)