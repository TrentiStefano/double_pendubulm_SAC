import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import re
from double_pendulum.model.symbolic_plant import SymbolicDoublePendulum
from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.simulation.simulation import Simulator
from stable_baselines3.common.monitor import Monitor
from double_pendulum.utils.wrap_angles import wrap_angles_diff, wrap_angles_top
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from Custom_Env import CustomEnv, double_pendulum_dynamics_func

script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, "log_SAC_pendubot")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#------------------- Setup environment -------------------#
model_par_path = os.path.join(script_dir, 'pendubot_parameters.yml')
torque_limit = [10.0, 0.0]
goal = np.array([np.pi, 0.0, 0.0, 0.0])
active_act = 0
mpar = model_parameters(filepath=model_par_path)
mpar.set_motor_inertia(0.0)
mpar.set_damping([0.0, 0.0])
mpar.set_cfric([0.0, 0.0])
mpar.set_torque_limit(torque_limit)
epsilon = 0.2
dt = 0.001
t_final = 5.0
integrator = "runge_kutta"
plant = SymbolicDoublePendulum(model_pars=mpar)
simulator = Simulator(plant=plant)

#learning environment
state_representation = 2
obs_space = gym.spaces.Box(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]))
act_space = gym.spaces.Box(np.array([-1]), np.array([1]))
termination = False

timepenalty = 0
previous_action = 0
in_roa = False
timesteps = 1

#initialize double pendulum dynamics
dynamics_func = double_pendulum_dynamics_func(
    simulator=simulator,
    dt=dt,
    integrator=integrator,
    robot="pendubot",
    state_representation=state_representation,
)
############################################################################
#------------------- SAC parameters -------------------#
max_steps = int(t_final / dt) #(5secondi)
n_envs = 100
training_steps = 8_000_000
verbose = 0
reward_threshold = 1_000_000
eval_freq=4000
n_eval_episodes=20
total_episodes = training_steps // max_steps
##############################################################################
#------------------- LQR ROA parameters -------------------#
rho = np.loadtxt(os.path.join(script_dir,"lqr_roa/rho"))
print("rho = ", rho)
S = np.loadtxt(os.path.join(script_dir,"lqr_roa/Smatrix"))
print("S matrix = ", S)

def check_if_state_in_roa(S, rho, x):
    xdiff = x - np.array([np.pi, 0.0, 0.0, 0.0])
    rad = np.einsum("i,ij,j", xdiff, S, xdiff)
    return rad < 0.5*rho, rad

###############################################################################
def calculate_energy(state, m1, m2, l1, l2, g=9.81):
    theta1, theta2, theta1_dot, theta2_dot = state
    theta1 = np.mod(theta1, 2 * np.pi)
    theta2 = np.mod(theta2 + np.pi, 2 * np.pi) - np.pi
    h1 = - (l1 / 2) * np.cos(theta1)
    h2 = - l1 * np.cos(theta1) - (l2 / 2) * np.cos(theta1 + theta2)
    #Potential energy
    PE = m1 * g * h1 + m2 * g * h2
    v1_sq = ((l1 / 2) * theta1_dot * np.sin(theta1))**2 + ((l1 / 2) * theta1_dot * np.cos(theta1))**2
    v2_sq = (l1 * theta1_dot * np.sin(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.sin(theta1 + theta2))**2 + \
            (l1 * theta1_dot * np.cos(theta1) + (l2 / 2) * (theta1_dot + theta2_dot) * np.cos(theta1 + theta2))**2
    I1 = (1/3) * m1 * l1**2
    I2 = (1/3) * m2 * l2**2
    # Kinetic energy
    KE = 0.5 * m1 * v1_sq + 0.5 * I1 * theta1_dot**2 + 0.5 * m2 * v2_sq + 0.5 * I2 * (theta1_dot + theta2_dot)**2
    
    return PE, KE

def reward_func(observation, action):
    global timepenalty, previous_action, timesteps, max_steps
    '''
    scale theta1: [-1, 1] -> [0, 2pi]
    scale theta2: [-1, 1] -> [-pi, +pi]
    '''
    s = dynamics_func.unscale_state(observation)
    
    W = np.diag([10, 10, 0.3, 0.3])
    state_error = s - goal
    state_distance_penalty = -1 * np.dot(state_error.T, np.dot(W ,state_error))
    torque_penalty = -0.0001 * np.dot(torque_limit[0]*action.T, torque_limit[0]*action)
    
    y_diff = wrap_angles_diff(s) #[-pi,pi]
    l1 = 0.4
    l2 = 0.1 
    h1 = - l1 * np.cos(y_diff[0])
    h2 = - l2 * np.cos(y_diff[0] + y_diff[1])
    ee_height = h1 + h2 #end effector height
        
    delta_action = np.abs(action[0] - previous_action)

    theta1_error = np.abs((np.pi - np.abs(y_diff[0])))/np.pi
    
    y = wrap_angles_top(s)
    m1 = 0.6
    m2 = 0.2
    Pe, Ke = calculate_energy(y, m1, m2, l1, l2, g=9.81)
    
    bonus,rad = check_if_state_in_roa(S,rho,y)
    
    r = state_distance_penalty + torque_penalty
    
    if ee_height >= 0.85*(l1+l2):
        r += 100*ee_height + 5*Pe
        if bonus:
            r += 1000
    else:
        r += - 0.01*torque_penalty -5*delta_action - 100*theta1_error
    
    previous_action = action[0]
    #print("total reward = ", r)
    return np.clip(r, -1e8,1e8)

def terminated_func(observation):
    s = dynamics_func.unscale_state(observation)
    y = wrap_angles_top(s)
    bonus,_ = check_if_state_in_roa(S,rho,y)
    if bonus:
        return True
    else:
        return False

def interpolate_start_position(progress, start_pos, goal_pos):
    return progress * start_pos + (1 - progress) * goal_pos

def noisy_reset_func():
    global timesteps, n_envs, total_episodes
    # Set the starting position (pendulum down: -1.0, 0.0) and the goal (pendulum up: 0.0, 0.0)
    start_pos = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Total training steps for full progress
    total_training_steps = 10_000_000
    progress = min(1.0, (timesteps//n_envs)*n_envs / total_training_steps)
    initial_pos = interpolate_start_position(progress, start_pos, goal_pos)
    noise = np.random.rand(4) * 0.01
    noise[2:] = noise[2:] - 0.05
    observation = initial_pos + noise
    
    #print(f"Progress: {progress}, Initial position: {initial_pos}, Noise: {noise}, Observation: {observation}")
    timesteps += 1
    
    return observation.astype(np.float32)

def zero_reset_func():
    observation = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return observation.astype(np.float32)

def get_latest_run_id(log_path):
    """
    Get the latest run id from TensorBoard logs by extracting the number after "SAC_".
    Increment the latest run id by 1 and return the new run name.
    """
    run_dirs = [d for d in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, d))]
    sac_run_dirs = [d for d in run_dirs if d.startswith("SAC_") and re.findall(r'\d+', d)]
    if len(sac_run_dirs) > 0:
        run_ids = [int(re.findall(r'\d+', d)[0]) for d in sac_run_dirs]
        latest_run_id = max(run_ids)
        new_run_id = latest_run_id + 1
    else:
        # If no SAC runs exist, start with SAC_1
        new_run_id = 1
    
    return f"SAC_{new_run_id}"

###############################################################################
#------------------- custom env -------------------#
env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=noisy_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,         
)
envs = make_vec_env(
    env_id=CustomEnv,
    n_envs=n_envs,
    env_kwargs={
        "dynamics_func": dynamics_func,
        "reward_func": reward_func,
        "terminated_func": terminated_func,
        "reset_func": noisy_reset_func,
        "obs_space": obs_space,
        "act_space": act_space,
        "max_episode_steps": max_steps,
    },
)
#------------------- evaluation env -------------------#
eval_env = CustomEnv(
    dynamics_func=dynamics_func,
    reward_func=reward_func,
    terminated_func=terminated_func,
    reset_func=zero_reset_func,
    obs_space=obs_space,
    act_space=act_space,
    max_episode_steps=max_steps,
)

eval_env = Monitor(eval_env)

tensorboard_log_path = os.path.join(log_dir, "tb_logs")
if not os.path.exists(tensorboard_log_path):
    os.makedirs(tensorboard_log_path)

def linear_schedule(initial_value, min_value=5e-4):
    def func(progress):
        # Linearly decay but ensure it's above the minimum value
        return max(min_value, initial_value * (1 - progress))
    return func

# Define the MLP architecture: 3 layers of 128 neurons each
policy_kwargs = dict(net_arch=[128, 128, 128])

#------------------- SAC anget -------------------#
agent = SAC(
    "MlpPolicy",
    envs,
    policy_kwargs=policy_kwargs,
    verbose=verbose,
    tensorboard_log=tensorboard_log_path,
    learning_rate=linear_schedule(0.01),
    ent_coef='auto_0.2',
    device = "cuda",
    seed = 0
)

run_name = get_latest_run_id(tensorboard_log_path)
best_model_save_path = os.path.join(log_dir, run_name)

#------------------- Training callbacks -------------------#
callback_on_best = StopTrainingOnRewardThreshold(
    reward_threshold=reward_threshold, verbose=verbose
)

eval_callback = EvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    best_model_save_path=best_model_save_path,
    log_path=log_dir,
    eval_freq=eval_freq,
    verbose=verbose,
    n_eval_episodes=n_eval_episodes,
)

checkpoint_callback = CheckpointCallback(
    save_freq=10_000, 
    save_path=best_model_save_path,
    name_prefix='SAC_checkpoint', 
    save_replay_buffer = True)

callback = CallbackList([checkpoint_callback, eval_callback])

#------------------- Training -------------------#
check_env(env)
if __name__ == '__main__':
    timesteps = 0
    agent.learn(total_timesteps=training_steps, callback=callback, progress_bar=True)
