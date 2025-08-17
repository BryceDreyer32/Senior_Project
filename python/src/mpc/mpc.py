# mpc_bldc_load.py
# Robust linear MPC with disturbance (friction) estimation for a geared BLDC driving a load.
# Requires: pip install numpy cvxpy osqp

import time
import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.realpath('python/src'))
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication
from hal.hal import HAL

fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
hal = HAL(fpga, None)

# ===================== User-configurable plant params ===================== #
dt = 0.02            # 20 ms loop
J  = 2.58648e-05     # kg·m^2 (motor inertia reflected to load); add load inertia if known
b  = 0.04            # N·m·s/rad (electrical damping baseline; rest handled by disturbance)
Kt = 0.025            # N·m per unit command at load (initial guess; tune)
u_max = 0.6
du_max = 0.05
w_max = 3.0

# Angle wrapping (12-bit encoder -> 0..4095); adjust CPR as needed
ENC_BITS = 12
ENC_CPR  = 1 << ENC_BITS
ENC_WRAP = 2 * math.pi

# ===================== MPC weights ===================== #
N = 20               # horizon steps
Q_th = 8.0           # angle error
Q_om = 6.0           # angular velocity
R_du = 0.60          # input rate penalty
R_u  = 0.01          # input effort penalty
R_d  = 1e-4          # discourage wild d changes inside horizon (kept small)
Qf   = 20.0          # terminal angle weight

# Kick heuristic for stiction
ENABLE_KICK   = True
KICK_THRESH_W = 0.05     # rad/s: "at rest"
KICK_MIN_ERR  = math.radians(1.5)  # only kick if >= ~1.5°
KICK_MAG      = 0.10      # extra on top of MPC solution, saturated by u_max
KICK_T_S      = 0.040     # seconds
KICK_STEPS    = max(1, int(KICK_T_S / dt))

MOTOR = 2

# ===================== I/O stubs (replace for hardware) ===================== #
def read_encoder_counts() -> int:
    """Return 0..4095 from the load encoder. Replace with your hardware read."""
    return hal.get_angle(MOTOR)

# User-provided I/O hooks you need to fill in:
def write_pwm_duty(duty: int, dir):
    """Send integer duty to your PWM peripheral."""
    if(duty == 0):
        hal.stop_motor(MOTOR)
    else:
        hal.run_motor(MOTOR, duty, dir)

def write_dir(bit: int):
    """Set direction GPIO: 0 or 1."""
    # TODO: implement hardware write
    pass

def set_pwm_from_u(u: float):
    """
    Map u in [-1, 1] to (dir_bit, duty in [20, 50]).
    u=0 -> motor idle (we'll emit duty=0 meaning 'off' if supported; else duty=20).
    """
    u = float(np.clip(u, -1.0, 1.0))
    mag = abs(u)

    # If you can fully stop PWM, do it below a tiny deadband:
    if mag < 1e-3:
        write_pwm_duty(0, 0)      # if 0 means 'off' on your hardware
        # If '0' is not valid, use the lowest allowed 'barely moves' value:
        # write_pwm_duty(20)
        # Also consider commanding a tiny "hold" direction if needed:
        return

    dir_bit = 1 if u >= 0 else 0
    #write_dir(dir_bit)

    DUTY_MIN = 20  # barely moves (given)
    DUTY_MAX = 38  # max safe (given)
    duty_span = DUTY_MAX - DUTY_MIN

    duty = int(round(DUTY_MIN + mag * duty_span))
    duty = max(DUTY_MIN, min(DUTY_MAX, duty))
    write_pwm_duty(duty, dir_bit)


# ===================== Helpers ===================== #
def enc_to_angle_rad(counts: int) -> float:
    """Map 0..CPR-1 to angle in [0, 2π)."""
    return (counts % ENC_CPR) * (ENC_WRAP / ENC_CPR)

def unwrap_angle(prev: float, new_wrapped: float) -> float:
    """Unwrap angle to a continuous angle (rad)."""
    dw = (new_wrapped - prev + math.pi) % (2*math.pi) - math.pi
    return prev + dw

class DerivativeLPF:
    """Finite-difference derivative with one-pole low-pass."""
    def __init__(self, dt, fc=10.0):
        self.dt = dt
        self.alpha = 2*math.pi*fc*dt / (1 + 2*math.pi*fc*dt)
        self.prev = None
        self.v = 0.0
    def update(self, x):
        if self.prev is None:
            self.prev = x
            return 0.0
        dx = (x - self.prev) / self.dt
        self.v = (1 - self.alpha)*self.v + self.alpha*dx
        self.prev = x
        return self.v

class LowPass:
    def __init__(self, dt, fc=3.0):
        self.alpha = 2*math.pi*fc*dt / (1 + 2*math.pi*fc*dt)
        self.y = 0.0
        self.init = False
    def update(self, x):
        if not self.init:
            self.y = x
            self.init = True
            return x
        self.y = (1 - self.alpha)*self.y + self.alpha*x
        return self.y

# ===================== Disturbance observer (very simple) ===================== #
class DisturbanceObserver:
    """
    Estimate d (load-side opposing torque) from measured (theta, omega) and applied u.
    Model: J*omega_dot = Kt*u - d - b*omega  =>  d = Kt*u - J*omega_dot - b*omega
    We low-pass the raw estimate to reduce noise. This runs outside the QP.
    """
    def __init__(self, dt, J, b, Kt, fc=3.0):
        self.dt = dt
        self.J = J
        self.b = b
        self.Kt = Kt
        self.d_lpf = LowPass(dt, fc=fc)
        self.om_diff = DerivativeLPF(dt, fc=20.0)
        self.prev_omega = None
    def update(self, omega, u):
        omega_dot = self.om_diff.update(omega)
        d_raw = self.Kt*u - self.J*omega_dot - self.b*omega
        return self.d_lpf.update(d_raw)

# ===================== Build MPC (single QP reused every step) ===================== #
class DisturbanceMPC:
    def __init__(self, dt, J, b, Kt, N, u_max, du_max, w_max=None,
                 Q_th=8.0, Q_om=1.0, R_du=0.15, R_u=0.01, R_d=1e-4, Qf=12.0):
        self.dt = dt
        self.N = N
        self.u_max = u_max
        self.du_max = du_max
        self.w_max = w_max

        # Discrete-time matrices for augmented state x=[theta, omega, d]
        A = np.array([
            [1.0, dt,         0.0],
            [0.0, 1.0 - (b/J)*dt, -(dt/J)],
            [0.0, 0.0,        1.0]
        ])
        B = np.array([
            [0.0],
            [(Kt/J)*dt],
            [0.0]
        ])
        self.A, self.B = A, B

        # Variables
        nx, nu = 3, 1
        self.x = cp.Variable((nx, N+1))
        self.u = cp.Variable((nu, N))
        self.du = cp.Variable((nu, N))
        self.dvar = self.x[2, :]  # disturbance trajectory (decision variable)

        # Parameters (updated each call)
        self.x0 = cp.Parameter(nx)
        self.ref = cp.Parameter(1)  # theta reference (rad)
        self.d0 = cp.Parameter(1)   # initial disturbance estimate

        # Constraints & Objective
        constr = []
        cost = 0

        # Initial conditions
        constr += [self.x[:, 0] == self.x0]

        # Dynamics + input rate definition
        for k in range(N):
            constr += [self.x[:, k+1] == A @ self.x[:, k] + B @ self.u[:, k]]
            if k == 0:
                # du_0 = u_0 - u_{-1}; we’ll set u_{-1} via a parameterized last_u
                pass
            else:
                constr += [self.du[:, k] == self.u[:, k] - self.u[:, k-1]]

            # Input constraints
            constr += [cp.abs(self.u[:, k]) <= u_max]
            if k > 0:
                constr += [cp.abs(self.du[:, k]) <= du_max]

            # Optional omega constraint
            if w_max is not None:
                constr += [cp.abs(self.x[1, k]) <= w_max]

            # Stage cost: angle tracking, omega, input rate/effort, small penalty on disturbance motion
            th_err = self.x[0, k] - self.ref
            cost += Q_th * cp.sum_squares(th_err) \
                  + Q_om * cp.sum_squares(self.x[1, k]) \
                  + R_u  * cp.sum_squares(self.u[:, k])
            if k > 0:
                cost += R_du * cp.sum_squares(self.u[:, k] - self.u[:, k-1])

            # “Gentle tether” on d to keep it from drifting wildly (convex)
            if k > 0:
                cost += R_d * cp.sum_squares(self.dvar[k] - self.dvar[k-1])

        # Terminal cost
        th_err_T = self.x[0, N] - self.ref
        cost += Qf * cp.sum_squares(th_err_T)
        if w_max is not None:
            constr += [cp.abs(self.x[1, N]) <= w_max]

        # Extra: bind initial disturbance state to latest estimate (keeps consistency)
        # We do it softly via a small quadratic cost rather than a hard equality (more robust).
        self.d_init_weight = 10.0
        cost += self.d_init_weight * cp.sum_squares(self.x[2, 0] - self.d0)

        # Last input (for du_0 constraint) as a parameter
        self.u_last = cp.Parameter(nu)
        constr += [self.u[:, 0] - self.u_last <= du_max]
        constr += [self.u_last - self.u[:, 0] <= du_max]

        # Problem
        self.prob = cp.Problem(cp.Minimize(cost), constr)

    def solve(self, theta, omega, d_est, theta_ref, u_last):
        self.x0.value = np.array([theta, omega, d_est])
        self.ref.value = np.array([theta_ref])
        self.d0.value  = np.array([d_est])
        self.u_last.value = np.array([u_last])

        # Solve with OSQP (fast, reliable for QPs)
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, max_iter=4000, eps_abs=1e-4, eps_rel=1e-4, verbose=False)
        except Exception as e:
            # Fallback default if solver hiccups
            return float(np.clip(0.0, -self.u_max, self.u_max)), False

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return float(np.clip(0.0, -self.u_max, self.u_max)), False

        u0 = float(self.u.value[0, 0])
        return float(np.clip(u0, -self.u_max, self.u_max)), True

# ===================== Controller runtime ===================== #
class Controller:
    def __init__(self):
        self.obs = DisturbanceObserver(dt, J, b, Kt, fc=3.0)
        self.mpc = DisturbanceMPC(dt, J, b, Kt, N, u_max, du_max, w_max, Q_th, Q_om, R_du, R_u, R_d, Qf)
        self.unwrap_prev = 0.0
        self.der_w = DerivativeLPF(dt, fc=20.0)
        self.u_last = 0.0
        self.kick_count = 0

    def step(self, theta_meas_wrapped, theta_ref):
        # Angle unwrap + velocity
        theta_wr = enc_to_angle_rad(theta_meas_wrapped)
        theta = unwrap_angle(self.unwrap_prev, theta_wr)
        self.unwrap_prev = theta
        omega = self.der_w.update(theta)

        # Disturbance estimate
        d_est = self.obs.update(omega, self.u_last)

        # Solve MPC
        u_cmd, ok = self.mpc.solve(theta, omega, d_est, theta_ref, self.u_last)

        # Optional stiction kick (only on first few steps of a move)
        if ENABLE_KICK:
            err = theta_ref - theta
            # bring error to shortest path (for near wraparound moves up to 180°)
            err = (err + math.pi) % (2*math.pi) - math.pi
            need_kick = (abs(omega) < KICK_THRESH_W) and (abs(err) > KICK_MIN_ERR) and (self.kick_count < KICK_STEPS)
            if need_kick:
                sign = 1.0 if err > 0 else -1.0
                u_cmd = np.clip(u_cmd + sign*KICK_MAG, -u_max, u_max)
                self.kick_count += 1
            else:
                # Reset when moving or error small
                if abs(err) < KICK_MIN_ERR or abs(omega) > KICK_THRESH_W:
                    self.kick_count = 0

        # Output
        set_pwm_from_u(u_cmd)
        self.u_last = u_cmd
        return {
            "theta": theta,
            "omega": omega,
            "u": u_cmd,
            "d_est": d_est,
            "mpc_ok": ok
        }

# ===================== Example main loop ===================== #
def run_closed_loop(target_deg=20.0, settle_band_deg=0.5, timeout_s=3.0, plot=True):
    ctrl = Controller()
    theta0 = enc_to_angle_rad(read_encoder_counts())
    ctrl.unwrap_prev = theta0

    theta_ref = theta0 + math.radians(target_deg)
    t0 = time.time()

    log_t, log_theta, log_ref, log_u, log_d = [], [], [], [], []

    while True:
        counts = read_encoder_counts()
        state = ctrl.step(counts, theta_ref)

        now = time.time() - t0
        log_t.append(now)
        log_theta.append(math.degrees(state["theta"] - theta0))
        log_ref.append(math.degrees(theta_ref - theta0))
        log_u.append(state["u"])
        log_d.append(state["d_est"])

        err = (theta_ref - state["theta"] + math.pi) % (2*math.pi) - math.pi
        if abs(math.degrees(err)) < settle_band_deg:
            set_pwm_from_u(0.0)
            break

        if now > timeout_s:
            set_pwm_from_u(0.0)
            break

        time.sleep(dt)

    if plot:
        fig, ax1 = plt.subplots()
        ax1.plot(log_t, log_theta, label="Angle (deg)")
        ax1.plot(log_t, log_ref, "--", label="Target (deg)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (deg)")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(log_t, log_u, "r", label="u (command)")
        ax2.plot(log_t, log_d, "g", label="d_est (torque)")
        ax2.set_ylabel("u / d_est")
        ax2.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    # Demo: move +20°, then back −20°
    run_closed_loop(+20.0)
    time.sleep(0.5)
    run_closed_loop(-20.0)
    time.sleep(1)
    hal.stop_motor(MOTOR)    

