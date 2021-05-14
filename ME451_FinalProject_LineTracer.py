import numpy as np
import matplotlib.pyplot as plt
import control.matlab as mt
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
	if theta<0: theta = theta + 2*np.pi
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

	# Add input noise
	# U += np.random.uniform(-1.0, 1.0, size=(2, 1))*25

	# Limit the maximum voltage input
	U = np.clip(U, -50, 50)
	return U

# Non-linear state-space representation
def ODE(t, X, Alpha):
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
# 1) Constants
r = 0.03 # [m]; radius of wheel
L = 0.1 # [m]; wheelbase
# 2) Time domain
T = 10
dt = 1e-1
t = np.arange(0, T+dt, dt)
# 3) Initial condition
Xd = np.array([[-1.5, -1, 0]]).T 				# Desired initial state vector [x, y, theta]
X0 = np.array([[-1.5-0.25, -1+0.25, 0+1]])		# Actual initial state vector [x, y, theta]

# Initial and final flat output
z0 = np.array([[-1.5, -1]]).T 			# Initial position [x0, y0]
zf = np.array([[1.5, 1]]).T 			# Final position [xf, yf]
zdot0 = np.array([[0.1, 0]]).T 			# Initial speed [vx0, vy0]
zdotf = np.array([[0.1, 0]]).T 			# Final speed [vx0, vy0]
##################################
# Feasible trajectory generation #
##################################
# Construct flat output system
Alpha = find_coeffs(z0, zdot0, zf, zdotf, T)
# Compute (Xd, Ud) for all time t
Xd_list = []
Ud_list = []
for time in t:
	Xd, Ud = feasible_trajectory(Alpha, time)
	Xd_list.append(Xd)
	Ud_list.append(Ud)
Xd_list = np.asarray(Xd_list)
Ud_list = np.asarray(Ud_list)
# Unpack Xd
xd = Xd_list[:, 0]
yd = Xd_list[:, 1]
thetad = Xd_list[:, 2]
# Unpack Ud
omega_Ld = Ud_list[:, 0]
omega_Rd = Ud_list[:, 1]

####################
# Controlled input #
####################
Q = 500*np.eye(3)
R = np.eye(2)

sol = solve_ivp(ODE, [0, T], X0.flatten(), t_eval=t, vectorized=True, args=(Alpha,))

# Plot
plt.figure(figsize=(6, 5))
# Plot trajectory
# plt.subplot(1, 2, 1)
plt.axis('scaled')
plt.plot(xd, yd, linestyle='--', linewidth=2.5, color='C3', label='Desired')
plt.plot(sol.y[0,:], sol.y[1,:], color='C0', label="Actual")
plt.scatter(sol.y[0,0], sol.y[1,0], marker='x', c='C7', s=40, label='Starting point')
plt.scatter(sol.y[0,-1], sol.y[1,-1], marker='o', facecolors='none', edgecolors='C7', s=40, label='Ending point')
plt.xlim(-2, 2); plt.ylim(-1.5, 1.5)
plt.xlabel('x position [m]'); plt.ylabel('y position [m]')
plt.legend(); plt.grid(); plt.title('Position')
# Plot angle
# plt.subplot(1, 2, 2)
# plt.plot(thetad, linestyle='--', linewidth=2.5, color='C3', label='Desired')
# plt.plot(sol.y[2,:], color='C0', label="Actual angle")
# plt.legend(); plt.grid(); plt.title('Desired angle')
plt.savefig('ME451_Final_noMotor_displaced.pdf')
plt.show()