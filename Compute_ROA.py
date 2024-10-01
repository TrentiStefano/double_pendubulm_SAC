import os
import yaml
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load pendubot parameters from YAML file
param_file = os.path.join(script_dir, "pendubot_parameters.yml")
with open(param_file, 'r') as f:
    params = yaml.safe_load(f)

# Extract parameters
I1 = params['I1']
I2 = params['I2']
b1 = params['b1']
b2 = params['b2']
cf1 = params['cf1']
cf2 = params['cf2']
g = params['g']
l1 = params['l1']
l2 = params['l2']
m1 = params['m1']
m2 = params['m2']
r1 = params['r1']
r2 = params['r2']
tl1 = params['tl1']
tl2 = params['tl2']

# System parameters for pendubot
A = np.array([[0, 1, 0, 0], 
              [0, -b1/I1, -m2*g*r2/I1, 0], 
              [0, 0, 0, 1], 
              [0, 0, (m1+m2)*g*r1/I2, -b2/I2]])

B = np.array([[0, 0], 
              [1/I1, 0], 
              [0, 0], 
              [0, 1/I2]])

# LQR tuning parameters
Q = 3.0 * np.diag([0.64, 0.64, 0.1, 0.1])
R = np.eye(2) * 0.82

# Compute LQR gain
P = scipy.linalg.solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R).dot(B.T.dot(P))

# Lyapunov function candidate: V(x) = x.T @ P @ x
# Estimate the ROA by checking where V(x) < constant, i.e., inside a level set
S = P
rho = 0.5  # You can try different rho values based on the desired ROA size

# Generate grid points to visualize the ROA
theta1_vals = np.linspace(-2*np.pi, 2*np.pi, 200)  # Increase to 200 points
theta2_vals = np.linspace(-2*np.pi, 2*np.pi, 200)

grid = np.meshgrid(theta1_vals, theta2_vals)
X = np.stack(grid, axis=-1)
roa_region = np.zeros_like(X[..., 0])

# Evaluate the Lyapunov function on the grid points
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        state = np.array([X[i, j, 0], 0, X[i, j, 1], 0])  # (theta1, dtheta1, theta2, dtheta2)
        V = state.T @ S @ state
        if V < rho:
            roa_region[i, j] = 1

# Plot the ROA ellipsoid
plt.contourf(theta1_vals, theta2_vals, roa_region, levels=[0, 0.5, 1], colors=['white', 'red'])

plt.title("ROA Estimate via Lyapunov Function")
plt.xlabel("Theta 1 [rad]")
plt.ylabel("Theta 2 [rad]")
plt.grid(True)
plt.savefig("ROA_estimate.png", format='png', dpi=300, bbox_inches='tight')  # Save as PNG with 300 DPI
plt.show()

for i in range(0, X.shape[0], 20):
    for j in range(0, X.shape[1], 20):
        state = np.array([X[i, j, 0], 0, X[i, j, 1], 0])  # (theta1, dtheta1, theta2, dtheta2)
        V = state.T @ S @ state
        #print(f"Theta1: {X[i,j,0]}, Theta2: {X[i,j,1]}, V: {V}")


# Save results
save_dir = os.path.join(script_dir, "lqr_roa")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("pendubot")
print("Q=", Q)
print("R=", R)
print("S=", S)
print("Estimated ROA volume approximation based on rho:", rho)
np.savetxt(os.path.join(save_dir, "Smatrix"), S)
np.savetxt(os.path.join(save_dir, "Kmatrix"), K)
np.savetxt(os.path.join(save_dir, "rho"), [rho])

# Save pendubot parameters
with open(os.path.join(save_dir, "pendubot_parameters.yml"), 'w') as f:
    yaml.dump(params, f)
