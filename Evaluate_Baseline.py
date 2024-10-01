import os
import numpy as np
from double_pendulum.controller.lqr.lqr_controller import LQRController
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from double_pendulum.controller.abstract_controller import AbstractController
from double_pendulum.controller.combined_controller import CombinedController
from double_pendulum.utils.wrap_angles import wrap_angles_top, wrap_angles_diff
from double_pendulum.controller.pid.point_pid_controller import PointPIDController
from double_pendulum.analysis.leaderboard import get_swingup_time

def switch_to_PID(x):
    y = wrap_angles_top(x)
    z = np.abs(y)-goal
    #print("z = ",z)
    if abs(z[0]) < 0.15 and abs(z[1]) < 0.15 :#and abs(z[2]) < 0.3 and abs(z[3]) < 0.3:
        return True  
    else:
        return False
    
class Baseline(AbstractController):
    def __init__(self, torque_limit=[10.0, 0.0], dt=0.01):
        super().__init__()
        self.torque_limit = torque_limit
        self.target_energy = self.calculate_energy(np.array([np.pi, 0, 0, 0]))
        self.k_e = 5.0
        
    def calculate_energy(self, x, m1=0.6, m2=0.2, l1=0.4, l2=0.1, g=9.81):
        theta1, theta2, theta1_dot, theta2_dot = wrap_angles_diff(x)
        theta1 = np.mod(theta1, 2 * np.pi)
        theta2 = np.mod(theta2 + np.pi, 2 * np.pi) - np.pi  
        h1 = - (l1 / 2) * np.cos(theta1)
        h2 = - l1 * np.cos(theta1) - (l2 / 2) * np.cos(theta1 + theta2)
        PE = m1 * g * h1 + m2 * g * h2
        v1_sq = ((l1 / 2) * theta1_dot * np.sin(theta1))**2 + ((l1 / 2) * theta1_dot * np.cos(theta1))**2
        v2_sq = (l1 * theta1_dot * np.sin(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2))**2 + \
                 (l1 * theta1_dot * np.cos(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2))**2
        I1 = (1/3) * m1 * l1**2
        I2 = (1/3) * m2 * l2**2
        KE = 0.5 * m1 * v1_sq + 0.5 * I1 * theta1_dot**2 + 0.5 * m2 * v2_sq + 0.5 * I2 * (theta1_dot + theta2_dot)**2

        total_energy = PE + KE
        
        return total_energy
    
    def get_control_output_(self, x, t=None):
        current_energy = self.calculate_energy(x)
        energy_error = self.target_energy - current_energy
        
        # Energy-based control input
        u = self.k_e * energy_error
        
        u1 = np.clip(u, -self.torque_limit[0], self.torque_limit[0])
        u2 = 0 
        
        return np.array([u1, u2])

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

#------------------- Baseline -------------------#
controller1 = Baseline(torque_limit=torque_limit, dt=dt)

#------------------- PID -------------------#
controller2 = PointPIDController(torque_limit=torque_limit, dt=dt)
controller2.set_parameters(Kp=10, Kd=1.0, Ki=2.0)

#------------------- LQR -------------------#
controller3 = LQRController(model_pars=mpar)
controller3.set_goal(goal)
Q = 3.0 * np.diag([0.65, 0.75, 0.1, 0.1])
R = np.eye(2) * 10000
controller3.set_cost_matrices(Q=Q, R=R)
controller3.set_parameters(failure_value=0.0, cost_to_go_cut=15000)

def condition1(t, x):
    return False

def condition2(t, x):
    is_in_roa = switch_to_PID(x)
    if is_in_roa:
        print("switching to PID....")
        return is_in_roa
    return is_in_roa

def condition3(t, x):
    y = wrap_angles_top(x)
    z = np.abs(y)-goal
    #print("z = ",z)
    if abs(z[0]) < 0.15 and abs(z[1]) < 0.15 and abs(z[2]) < 0.4 and abs(z[3]) < 0.7:
        print("switching to LQR....")
        return True
    else:
        return False

stabilizing_controller = CombinedController(
    controller1=controller2,
    controller2=controller3,
    condition1=condition1,
    condition2=condition3,
    compute_both=False
)

controller = CombinedController(
    controller1=controller1,
    controller2=stabilizing_controller,
    condition1=condition1,
    condition2=condition2,
    compute_both=False
)
controller.init()
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
cumulative_reward = 0
# Compute cumulative reward
for t, x, u in zip(T, X, U):
    reward = reward_func(x, u)
    cumulative_reward += reward

print('Swingup time: ' + str(get_swingup_time(T, np.array(X), mpar=mpar)))
print('Total Cumulative Reward: ' + str(cumulative_reward))