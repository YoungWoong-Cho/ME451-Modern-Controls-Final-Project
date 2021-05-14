import numpy as np
import matplotlib.pyplot as plt
import control.matlab as mt
from tqdm import tqdm
from scipy.integrate import solve_ivp

# Basis functions
def lamb(t):
	return np.array([[t**3, t**2, t, 1]]).T
def lambdot(t):
	return np.array([[3*t**2, 2*t, 1, 0]]).T
def lambddot(t):
	return np.array([[6*t, 2, 0, 0]]).T

# Find coefficients for basis function
def find_coeffs(z0, zdot0, zf, zdotf, T):
	Y = np.concatenate((z0, zdot0, zf, zdotf), axis=1)
	B = np.concatenate((lamb(0), lambdot(0), lamb(T), lambdot(T)), axis=1)
	Alpha = Y@np.linalg.inv(B)
	return Alpha

# Generate feasible trajectory using the coefficient matrix at given time t
def feasible_trajectory(Alpha, t):
	# Obtain basis functions
	z1, z2 = Alpha@lamb(t)
	z1dot, z2dot = Alpha@lambdot(t)
	z1ddot, z2ddot = Alpha@lambddot(t)

	# Derive desired X
	x = z1 								# x position
	y = z2 								# y position
	theta = np.arctan2(z2dot, z1dot)	# angle
	Xd = np.array([x, y, theta])		# Desired state vector

	# Derive desired U
	v = np.sqrt(z1dot**2+z2dot**2)
	thetadot = (z1dot*z2ddot - z2dot*z1ddot)/(z1dot**2+z2dot**2)
	omega_L = (1/r)*(v-(L/2)*thetadot)		# Angular speed of left DC motor
	omega_R = (1/r)*(v+(L/2)*thetadot)		# Angular speed of right DC motor
	Ud = np.array([omega_L, omega_R])

	return (Xd, Ud)

# Compute input at given (X, U) with controller K
def control_input(Alpha, Q, R, X, t):
	# Generate feasible trajectory
	Xd, Ud = feasible_trajectory(Alpha, t)
	x, y, theta = Xd
	omega_L, omega_R = Ud
	x = float(x)
	y = float(y)
	theta = float(theta)
	omega_L = float(omega_L)
	omega_R = float(omega_R)

	# Find controller using LQR at time t
	A = np.array([[0, 0, -r/2*np.sin(theta)*(omega_L+omega_R)],
				  [0, 0, r/2*np.cos(theta)*(omega_L+omega_R)],
				  [0, 0, 0]])
	B = np.array([[r/2*np.cos(theta), r/2*np.cos(theta)],
				 [r/2*np.sin(theta), r/2*np.sin(theta)],
				 [-r/L, r/L]])

	K, _, _ = mt.lqr(A, B, Q, R)

	# Compute control law
	U = Ud - K@(X - Xd)
	return U

# Non-linear state-space representation
def ODE(t, X, Alpha):
	# print(X)
	x, y, theta = X
	omega_L, omega_R = control_input(Alpha, Q, R, X, t)		# Get U from control law
	xdot = r/2*np.cos(theta)*omega_L + r/2*np.cos(theta)*omega_R
	ydot = r/2*np.sin(theta)*omega_L + r/2*np.sin(theta)*omega_R
	thetadot = -r/L*omega_L + r/L*omega_R
	Xdot = np.concatenate((xdot, ydot, thetadot), axis=0)
	return Xdot

############################
# Simulation configuration #
############################
# 1) Dimension
r = 0.03 			# [m]; Radius of wheel
L = 0.1 			# [m]; Wheelbase
r_traj = 1      	# [m]; Radius of circular path
c = 2*np.pi*r_traj 	# [m]; Circimference
s = 0.02        	# [m]; Distance from sensors to axle
steps = int(c/s)//4

# 2) Domain
# Time domain
T = 1
dt = 1e-1
t = np.arange(0, T+dt, dt)
# Angle domain
angles = np.linspace(0, 2*np.pi, steps)-3*np.pi/2	# Angle domain, ranging from -3*pi/2 to pi/2

# 3) Initial condition
Xd = np.array([[0, r_traj, -np.pi]]).T 			# Desired state vector. Currently contains X0_d only
X = np.array([[-0.3, r_traj+0.2, 0]]).T 		# Actual state vector. Currently contains X0 only

# 4) LQR settings
Q = np.diag([10, 10, 1]) 	# Used 10 for x and y position and 1 for angle so that the priority is on position over angle
R = np.eye(2)

#############
# Simulate! #
#############
for idx in tqdm(range(len(angles)-1)):
	# Initial condition (for this iteration)
	X0 = X[:, -1]
	##################################
	# Feasible trajectory generation #
	##################################
	# Construct flat output system
	v_traj = r_traj*(angles[idx+1]-angles[idx])/T 											# Speed of rotation
	z0 = np.array([[r_traj*np.cos(angles[idx]), r_traj*np.sin(angles[idx])]]).T 			# Initial position [x0, y0]
	zf = np.array([[r_traj*np.cos(angles[idx+1]), r_traj*np.sin(angles[idx+1])]]).T 		# Final position [xf, yf]
	zdot0 = np.array([[-v_traj*np.sin(angles[idx]), v_traj*np.cos(angles[idx])]]).T 		# Initial speed [vx0, vy0]
	zdotf = np.array([[-v_traj*np.sin(angles[idx+1]), v_traj*np.cos(angles[idx+1])]]).T 	# Final speed [vx0, vy0]
	Alpha = find_coeffs(z0, zdot0, zf, zdotf, T)
	# Compute (Xd, Ud) for all time t
	for t_ in t:
		Xd_, _ = feasible_trajectory(Alpha, t_)
		Xd = np.concatenate((Xd, Xd_), axis=1)
	####################
	# Controlled input #
	####################
	sol = solve_ivp(ODE, [0, T], X0.flatten(), t_eval=t, vectorized=True, args=(Alpha,))
	X = np.concatenate((X, sol.y), axis=1)

# Plot
plt.figure(figsize=(12, 6))
# Plot position
plt.subplot(1, 2, 1)
plt.axis('scaled')
plt.plot(X[0,:], X[1,:], label='Actual', color='C0')
plt.plot(Xd[0,:], Xd[1,:], linestyle='--', label='Desired', color='C3')
plt.scatter(X[0,0], X[1,0], marker='x', c='C7', s=50, label='Starting point')
plt.scatter(X[0,-1], X[1,-1], marker='o', facecolors='none', edgecolors='C7', s=50, label='Ending point')
plt.xlim(-r_traj-0.5, r_traj+0.5); plt.ylim(-r_traj-0.5, r_traj+0.5)
plt.xlabel('x position [m]'); plt.ylabel('y position [m]')
plt.legend(); plt.grid(); plt.title('Position')
# Plot angle
plt.subplot(1, 2, 2)
plt.plot(X[2,:], label='Actual', color='C0')
plt.plot(Xd[2,:], linestyle='--', label='Desired', color='C3')
plt.xlabel('Iteration'); plt.ylabel('Angle [rad]')
plt.legend(); plt.grid(); plt.title('Angle')
plt.show()