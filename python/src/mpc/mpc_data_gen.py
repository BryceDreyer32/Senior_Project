# mpc_data_gen.py
import numpy as np
import cvxpy as cp

def approx_sign(x, epsilon=0.01):
    return x / (cp.abs(x) + epsilon)

def approx_sign_linear(x, limit=0.01):
    return cp.maximum(cp.minimum(x / limit, 1), -1)

# ----- Motor + friction model -----
def motor_dynamics(theta, omega, u, dt=0.01):
    J = 1.309405e-06      # inertia
    b = 20e-6             # viscous friction
    tau_c = 260e-6        # Coulomb friction
    sign = np.sign(omega) if np.abs(omega) > 1e-3 else 0
    tau_friction = tau_c * sign + b * omega + 0.005 * omega**3
    alpha = (u - tau_friction) / J
    omega_new = omega + alpha * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new

# ----- MPC parameters -----
N = 10       # horizon
dt = 0.01
u_max = 0.5

# ----- Generate dataset -----
data_X, data_y = [], []

for _ in range(2000):  # scenarios
    theta, omega = np.random.uniform(-0.1, 0.1), 0.0
    theta_target = np.random.uniform(-0.1, 0.1)

    # MPC optimization
    u = cp.Variable(N)
    th = cp.Variable(N+1)
    om = cp.Variable(N+1)

    cost = 0
    constr = []
    constr += [th[0] == theta, om[0] == omega]

    for k in range(N):
        # discrete-time dynamics (linearized for MPC)
        constr += [th[k+1] == th[k] + dt * om[k]]
#        constr += [om[k+1] == om[k] + dt * (u[k] - 0.02*cp.sign(om[k]) - 0.001*om[k])]
#        constr += [om[k+1] == om[k] + dt * (u[k] - 0.02*approx_sign(om[k]) - 0.001*om[k])]
#        constr += [om[k+1] == om[k] + dt * (u[k] - 0.02 * approx_sign_linear(om[k]) - 0.001 * om[k])]
        constr += [om[k+1] == om[k] + dt * (u[k] - 0.001 * om[k])]

        constr += [cp.abs(u[k]) <= u_max]
        cost += 50 * cp.square(th[k] - theta_target) + 0.1 * cp.square(u[k])

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if prob.status != cp.OPTIMAL:
        continue

    # Save first-step control as label
    data_X.append([theta, omega, theta_target])
    data_y.append(u.value[0])

np.savez("python/src/mpc/mpc_dataset.npz", X=np.array(data_X), y=np.array(data_y))
print("Dataset saved:", len(data_X), "samples")

