import numpy as np
import sympy
from sympy import lambdify
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D

# TWO DIMENSIONAL CFD EQUATIONS
# discretize with Forward Difference in time
# discretize with Backward Difference in space
	
# u is velocity, x-velocity
# v is y-velocity
# t is time
# x,y is position
# c is wavespeed
# nu is viscosity

# STEP 5 - 2D LINEAR CONVECTION	
def linear_convection_2D(X, Y, Z, T, C):
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	c = C				# wavespeed
	
	# Starting velocity
	u = Z.copy()

	# Loop over time and space
	U = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		for j in range(1, nx):
			for i in range(1, ny):
				u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
									  (c * dt / dy * (un[j, i] - un[j - 1, i])))
								  
		u[0, :] = 1
		u[-1, :] = 1
		u[:, 0] = 1
		u[:, -1] = 1
			
		U.append(u.copy())
	
	return np.array(U)
	
# STEP 6 - 2D NON-LINEAR CONVECTION	
def nonlinear_convection_2D(X, Y, Zx, Zy, T):
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = Zx.copy()
	v = Zy.copy()
	w = Zy.copy()

	# Loop over time and space
	W = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		vn = v.copy()
		for j in range(1, nx):
			for i in range(1, ny):
				u[j, i] = (un[j, i] - (un[j, i] * dt / dx * (un[j, i] - un[j, i - 1])) -
									  (vn[j, i] * dt / dy * (un[j, i] - un[j - 1, i])))
				v[j, i] = (vn[j, i] - (un[j, i] * dt / dx * (vn[j, i] - vn[j, i - 1])) -
									  (vn[j, i] * dt / dy * (vn[j, i] - vn[j - 1, i])))
				w[j, i] = np.sqrt(u[j,i]**2 + v[j,i]**2)
								  
		u[0, :] = 1
		u[-1, :] = 1
		u[:, 0] = 1
		u[:, -1] = 1
		
		v[0, :] = 1
		v[-1, :] = 1
		v[:, 0] = 1
		v[:, -1] = 1
			
		W.append(w.copy())
	
	return np.array(W)
	
	
# STEP 7 - 2D DIFFUSION
def diffusion_2D(X, Y, Z, T, nu):
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = Z.copy()

	# Loop over time and space
	U = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		for j in range(1, nx-1):
			for i in range(1, ny-1):
				u[j, i] = (un[j, i] + (nu * dt / dx**2 * (un[j, i+1] - 2*un[j, i] + un[j, i - 1])) +
									  (nu * dt / dy**2 * (un[j+1, i] - 2*un[j, i] + un[j - 1, i])))
								  
		u[0, :] = 1
		u[-1, :] = 1
		u[:, 0] = 1
		u[:, -1] = 1
			
		U.append(u.copy())
	
	return np.array(U)


# STEP 8 - 2D BURGERS' EQUATION	
def burgers_equation_2D(X, Y, Zx, Zy, T, nu):
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = Zx.copy()
	v = Zy.copy()
	w = Zy.copy()

	# Loop over time and space
	W = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		vn = v.copy()
		for j in range(1, nx-1):
			for i in range(1, ny-1):
				u[j, i] = (un[j, i] - (un[j, i] * dt / dx * (un[j, i] - un[j, i - 1])) -
									  (vn[j, i] * dt / dy * (un[j, i] - un[j - 1, i])) +
									  (nu * dt / dx**2 * (un[j, i+1] - 2*un[j, i] + un[j, i - 1])) +
									  (nu * dt / dy**2 * (un[j+1, i] - 2*un[j, i] + un[j - 1, i])))
				v[j, i] = (un[j, i] - (un[j, i] * dt / dx * (vn[j, i] - vn[j, i - 1])) -
									  (vn[j, i] * dt / dy * (vn[j, i] - vn[j - 1, i])) +
									  (nu * dt / dx**2 * (vn[j, i+1] - 2*vn[j, i] + vn[j, i - 1])) +
									  (nu * dt / dy**2 * (vn[j+1, i] - 2*vn[j, i] + vn[j - 1, i])))
				w[j, i] = np.sqrt(u[j,i]**2 + v[j,i]**2)
								  
		u[0, :] = 1
		u[-1, :] = 1
		u[:, 0] = 1
		u[:, -1] = 1
		
		v[0, :] = 1
		v[-1, :] = 1
		v[:, 0] = 1
		v[:, -1] = 1
			
		W.append(w.copy())
	
	return np.array(W)
	
def velocity_function0(X, Y):
	Z = np.ones([len(X), len(Y)])
	for i in np.arange(len(X)):
		for j in np.arange(len(Y)):
			if 0.1 < X[i] and X[i] < 0.6 and 0.2 < Y[j] and Y[j] < 0.7:
				Z[i,j] = 2
			
	return Z	

def velocity_function1(X, Y):
	Z = np.ones([len(X), len(Y)])
	for i in np.arange(len(X)):
		for j in np.arange(len(Y)):
			if 0.3 < X[i] and X[i] < 0.9 and 0.7 < Y[j] and Y[j] < 0.7:
				Z[i,j] = 2
			
	return Z


	
# EQUATION RUNNER
################################################################################################################

Interval = 1
C = 0.1
nu = 0.05
X = np.linspace(0, 2, 31)
Y = np.linspace(0, 2, 31)
T = np.linspace(0, 1, 181)
Zx = np.array(velocity_function0(X,Y))
Zy = np.array(velocity_function1(X,Y))

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim3d(1, 2.2)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')       
xx, yy = np.meshgrid(X, Y)

#ness = linear_convection_2D(X, Y, Zx, T, C)
#ness = nonlinear_convection_2D(X, Y, Zx, Zy, T)
#ness = diffusion_2D(X, Y, Zx, T, nu)
ness = burgers_equation_2D(X, Y, Zx, Zy, T, nu)


surf = None
def animate_2D(i):
	global surf
	if surf:
		ax.collections.remove(surf)
		
	surf = ax.plot_surface(xx, yy, ness[i], cmap=cm.viridis, rstride=2, cstride=2)

anim = animation.FuncAnimation(fig, animate_2D, frames=ness.shape[0], interval=Interval)

plt.rcParams['animation.ffmpeg_path'] = './ffmpeg'
mywriter = animation.FFMpegWriter(fps=60)
anim.save('burgers_equation.mp4',writer=mywriter)

plt.show()
	