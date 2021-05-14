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
def lambdddot(t):
	return np.array([[6, 0, 0, 0]]).T

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
	z1dddot, z2dddot = Alpha@lambdddot(t)

	# Derive desired X
	x = z1 								# x position
	y = z2 								# y position
	theta = np.arctan2(z2dot, z1dot)	# angle
	if theta>np.pi/2: theta -= 2*np.pi 	# to prevent angle flip
	thetadot = (z1dot*z2ddot - z2dot*z1ddot)/(z1dot**2+z2dot**2)
	thetaddot = np.nan_to_num(((2*z2dot*z1ddot**2/z1dot**3) - (2*z2ddot*z1ddot/z1dot**2) - (z1dddot*z2dot/z1dot**2) + (z2dddot/z1dot)) / (z2dot**2/z1dot**2 + 1) \
				- (((z2ddot/z1dot) - (z2dot*z1ddot/z1dot**2))*((2*z2dot*z2ddot/z1dot**2) - (2*z2dot**2*z1ddot/z1dot**3))) / (z2dot**2/z1dot**2 + 1)**2)
	v = np.sqrt(z1dot**2+z2dot**2)
	vdot = (z1dot*z1ddot+z2dot*z2ddot)/np.sqrt(z1dot**2+z2dot**2)
	omega_L = (1/r)*(v-(L/2)*thetadot)				# Angular speed of left DC motor
	omega_Ldot = (1/r)*(vdot-(L/2)*thetaddot)		# Angular acceleration of left DC motor
	omega_R = (1/r)*(v+(L/2)*thetadot)				# Angular speed of right DC motor
	omega_Rdot = (1/r)*(vdot+(L/2)*thetaddot)		# Angular acceleration of right DC motor
	Xd = np.array([x, y, theta, omega_L, omega_R])	# Desired state vector

	# Derive desired U
	V_L = tau/K_m*(omega_Ldot + omega_L)	# Voltage to left DC motor
	V_R = tau/K_m*(omega_Rdot + omega_R)	# Voltage to right DC motor
	Ud = np.array([V_L, V_R])

	return Xd, Ud

# Compute input at given (X, U) with controller K
def control_input(Alpha, Q, R, X, t):
	# Generate feasible trajectory
	Xd, Ud = feasible_trajectory(Alpha, t)
	x, y, theta, omega_L, omega_R = Xd
	V_L, V_R = Ud
	x = float(x)
	y = float(y)
	theta = float(theta)
	omega_L = float(omega_L)
	omega_R = float(omega_R)

	# Find controller using LQR at time t
	# Combine a transfer function from Voltage to omega
	# with ss representation from omega to state
	A = np.array([[0, 0, -r/2*np.sin(theta)*(omega_L+omega_R), r/2*np.cos(theta), r/2*np.cos(theta)],
				  [0, 0, r/2*np.cos(theta)*(omega_L+omega_R), r/2*np.sin(theta), r/2*np.sin(theta)],
				  [0, 0, 0, -r/L, r/L],
				  [0, 0, 0, -1/tau, 0],
				  [0, 0, 0, 0, -1/tau]])
	B = np.array([[0, 0],
				  [0, 0],
				  [0, 0],
				  [K_m/tau, 0],
				  [0, K_m/tau]])
	
	# Check controllability and find controller using LQR only if controllable
	AB = np.zeros((A.shape[0], B.shape[1]*(A.shape[0]-1)))
	for i in range(1, A.shape[0]):
		AB[:,i:i+2] = np.linalg.matrix_power(A, i)@B
	ctrb = np.hstack((B,AB))
	rank_ctrb = np.sum(np.linalg.svd(ctrb, compute_uv=False) > 1e-10)
	K = np.zeros((2, 5))
	if rank_ctrb == 5:
		K, _, _ = mt.lqr(A, B, Q, R)

	# Compute control law
	U = Ud - K@(X - Xd)

	# Add input noise
	# U += np.random.uniform(-2.0, 2.0, size=(2, 1))

	# Limit the maximum voltage input
	U = np.clip(U, -12, 12)
	return U

# Non-linear state-space representation
def ODE(t, X, Alpha):
	x, y, theta, omega_L, omega_R = X
	V_L, V_R = control_input(Alpha, Q, R, X, t)		# Get U from control law
	xdot = r/2*np.cos(theta)*omega_L + r/2*np.cos(theta)*omega_R
	ydot = r/2*np.sin(theta)*omega_L + r/2*np.sin(theta)*omega_R
	thetadot = -r/L*omega_L + r/L*omega_R
	omegaLdot = np.asarray((K_m*V_L-omega_L)/tau).reshape(1,)
	omegaRdot = np.asarray((K_m*V_R-omega_R)/tau).reshape(1,)
	Xdot = np.concatenate((xdot, ydot, thetadot, omegaLdot, omegaRdot), axis=0)
	return Xdot

############################
# Simulation configuration #
############################
# 1) Dimension
r = 0.03 			# [m]; Radius of wheel
L = 0.1 			# [m]; Wheelbase
a = 1
c = 6.09722*a 		# approximate curve length
s = 0.02			# [m]; Distance from sensors to axle
steps = int(c/s)
K_m = 4.9			# [rad/V/s];
tau = 3				# [s];


# 2) Domain
# Time domain
T = 1
dt = 1e-1
t = np.arange(0, T+dt, dt)
# Angle domain
angles = np.linspace(0, 2*np.pi, steps)		# Angle domain, ranging from -3*pi/2 to pi/2

# 3) Initial condition
Xd = np.array([[0, 0, np.pi/4, 0, 0]]).T 			# Desired state vector. Currently contains X0_d only
X = np.array([[0, 0, np.pi/4, 0, 0]]).T 	# Actual state vector. Currently contains X0 only

# 4) LQR settings
Q = np.diag([100, 100, 10, 1, 1])
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
	v_traj = a*(angles[idx+1]-angles[idx])/T	# Speed of rotation
	z0 = a*np.array([[np.sin(angles[idx]), np.sin(angles[idx])*np.cos(angles[idx])]]).T 			# Initial position [x0, y0]
	zf = a*np.array([[np.sin(angles[idx+1]), np.sin(angles[idx+1])*np.cos(angles[idx+1])]]).T 		# Final position [xf, yf]
	zdot0 = v_traj*np.array([[np.cos(angles[idx]), np.cos(angles[idx])**2-np.sin(angles[idx])**2]]).T 		# Initial speed [vx0, vy0]
	zdotf = v_traj*np.array([[np.cos(angles[idx+1]), np.cos(angles[idx+1])**2-np.sin(angles[idx+1])**2]]).T 		# Final speed [vxf, vyf]
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
plt.figure(figsize=(15, 10))
# Plot position
plt.subplot(2, 2, 1)
plt.axis('scaled')
plt.plot(X[0,:], X[1,:], label='Actual', color='C0')
plt.plot(Xd[0,:], Xd[1,:], linestyle='--', label='Desired', color='C3')
plt.scatter(X[0,0], X[1,0], marker='x', c='C7', s=50, label='Starting point')
plt.scatter(X[0,-1], X[1,-1], marker='o', facecolors='none', edgecolors='C7', s=50, label='Ending point')
plt.xlim(-a-0.5, a+0.5); plt.ylim(-a-0.5, a+0.5)
plt.xlabel('x position [m]'); plt.ylabel('y position [m]')
plt.legend(); plt.grid(); plt.title('Position')
# Plot angle
plt.subplot(2, 2, 2)
plt.plot(X[2,:], label='Actual', color='C0')
plt.plot(Xd[2,:], linestyle='--', label='Desired', color='C3')
plt.xlabel('Iteration'); plt.ylabel('Angle [rad]')
plt.legend(); plt.grid(); plt.title('Angle')
# Plot x
plt.subplot(4, 2, 5)
plt.plot(X[0,:], label='Actual x', color='C0')
plt.plot(Xd[0,:], linestyle='--', label='Desired x', color='C3')
plt.legend(); plt.grid()
# Plot y
plt.subplot(4, 2, 7)
plt.plot(X[1,:], label='Actual y', color='C0')
plt.plot(Xd[1,:], linestyle='--', label='Desired y', color='C3')
plt.legend(); plt.grid()
# Plot omega_L
plt.subplot(4, 2, 6)
plt.plot(X[3,:], label='$Actual \omega_{L}$', color='C0')
plt.plot(Xd[3,:], linestyle='--', label='$Desired \omega_{L}$', color='C3')
plt.legend(); plt.grid()
# Plot omega_R
plt.subplot(4, 2, 8)
plt.plot(X[4,:], label='$Actual \omega_{R}$', color='C0')
plt.plot(Xd[4,:], linestyle='--', label='$Desired \omega_{R}$', color='C3')
plt.legend(); plt.grid()
plt.show()