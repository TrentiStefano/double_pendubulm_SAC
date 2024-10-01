import os
import re
import numpy as np
from stable_baselines3 import SAC
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_diff, wrap_angles_top
from double_pendulum.analysis.leaderboard import get_swingup_time

from Custom_Env import CustomEnv, double_pendulum_dynamics_func

class SACController(AbstractController):
    def __init__(self, model_path, dynamics_func, dt):
        super().__init__()
        self.model = SAC.load(model_path)
        self.dynamics_func = dynamics_func
        self.dt = dt

    def get_control_output_(self, x, t=None):
        obs = self.dynamics_func.normalize_state(x)
        action = self.model.predict(obs)
        #print("action", action)
        u = self.dynamics_func.unscale_action(np.asarray(action[0]))
        return u

###############################################################################

# Previous action and torque limit setup
previous_action = np.zeros(1)
torque_limit = [10.0, 0.0]

# Reward function based on provided energy calculation
def calculate_energy(state, m1, m2, l1, l2, g=9.81):
    theta1, theta2, theta1_dot, theta2_dot = state
    theta1 = np.mod(theta1, 2 * np.pi)
    theta2 = np.mod(theta2 + np.pi, 2 * np.pi) - np.pi
    h1 = - (l1 / 2) * np.cos(theta1)
    h2 = - l1 * np.cos(theta1) - (l2 / 2) * np.cos(theta1 + theta2)
    # Potential energy
    PE = m1 * g * h1 + m2 * g * h2
    v1_sq = ((l1 / 2) * theta1_dot * np.sin(theta1))**2 + ((l1 / 2) * theta1_dot * np.cos(theta1))**2
    v2_sq = (l1 * theta1_dot * np.sin(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2))**2 + \
            (l1 * theta1_dot * np.cos(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2))**2
    I1 = (1/3) * m1 * l1**2
    I2 = (1/3) * m2 * l2**2
    # Kinetic energy
    KE = 0.5 * m1 * v1_sq + 0.5 * I1 * theta1_dot**2 + 0.5 * m2 * v2_sq + 0.5 * I2 * (theta1_dot + theta2_dot)**2
    
    return PE, KE

def reward_func(observation, action, goal=np.array([np.pi, 0.0, 0.0, 0.0])):
    global previous_action
    action = np.array(action)
    s = observation
    W = np.diag([10, 10, 0.3, 0.3])
    state_error = s - goal
    state_distance_penalty = -1 * np.dot(state_error.T, np.dot(W, state_error))
    torque_penalty = -0.0001 * np.dot(torque_limit[0] * action, torque_limit[0] * action)
    r = state_distance_penalty + torque_penalty
    y_diff = wrap_angles_diff(s)
    l1 = 0.4
    l2 = 0.1
    h1 = - l1 * np.cos(y_diff[0])
    h2 = - l2 * np.cos(y_diff[0] + y_diff[1])
    ee_height = h1 + h2  # end-effector height
    delta_action = np.abs(action[0] - previous_action)
    theta1_error = np.abs((np.pi - np.abs(y_diff[0]))) / np.pi
    y = wrap_angles_top(s)
    m1 = 0.6
    m2 = 0.2
    Pe, Ke = calculate_energy(y, m1, m2, l1, l2, g=9.81)
    if ee_height >= 0.85 * (l1 + l2):
        r += 100 * ee_height + 5 * Pe
    else:
        r += -0.01 * torque_penalty - 5 * delta_action - 100 * theta1_error

    previous_action = action[0]
    
    return np.clip(r, -1e8, 1e8)

############################################################################### 
# Create the plant and simulator for the Pendubot
script_dir = os.path.dirname(os.path.abspath(__file__))
model_par_path = os.path.join(script_dir, 'pendubot_parameters.yml')
torque_limit = [10.0, 0.0]
dt = 0.01
integrator = "runge_kutta"
goal = np.array([np.pi, 0.0, 0.0, 0.0])
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot="pendubot",
    state_representation=2,
)

#------------------- SAC agent -------------------#
latest_model_path = os.path.join(script_dir, 'SAC_agent/best_model.zip')

controller1 = SACController(
    model_path=latest_model_path,
    dynamics_func=dynamics_func,
    dt=dt
)

#------------------- LQR -------------------#
rho = np.loadtxt("Test/lqr_roa/rho")
S = np.loadtxt("Test/lqr_roa/Smatrix")
Q = 3.0 * np.diag([0.64, 0.64, 0.1, 0.1])
R = np.eye(2) * 0.82
controller2 = LQRController(model_pars=mpar)
controller2.set_goal(goal)
controller2.set_cost_matrices(Q=Q, R=R)
controller2.set_parameters(failure_value=0.0,cost_to_go_cut=50)

def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < 0.5*rho, rad

def condition1(t, x):
    return False

def condition2(t, x):
    y = wrap_angles_diff(x)
    flag,rad = check_if_state_in_roa(S,rho,y)
    if flag: 
        return flag
    return flag

#------------------- Combined controller -------------------#
controller = CombinedController(
    controller1=controller1,
    controller2=controller2,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()

#------------------- Simulation -------------------#
T, X, U = simulator.simulate_and_animate(
    t0=0.0,
    x0=[0.0, 0.0, 0.0, 0.0],
    tf=30.0,
    dt=dt,
    controller=controller,
    integrator=integrator,
    save_video=False,
)

# Initialize cumulative reward
cumulative_reward = 0.0
for i in range(len(T)):
    obs = X[i]
    if i < len(U):
        action = U[i]
    else:
        action = [0.0, 0.0]
    reward = reward_func(obs, action)
    cumulative_reward += reward

    # Output results
print(f"Cumulative reward: {cumulative_reward}")
print('Swingup time: ' + str(get_swingup_time(T, np.array(X), mpar=mpar)))