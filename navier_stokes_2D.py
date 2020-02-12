import numpy as np
import sympy
from sympy import lambdify
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D

# TWO DIMENSIONAL CFD EQUATIONS
# discretize with Forward Difference in time
# discretize with Backward Difference in space
	
# u is velocity
# t is time
# x,y is position
# c is wavespeed
# nu is viscosity

# STEP 1 - 2D LINEAR CONVECTION	
def linear_convection_2D(X, Y, V, T, C):
	# du/dt + c * [du/dx] + c * [du/dy] = 0
	# solution: u_(i,j)_(n+1) = u_(i,j)_(n) - c * [dt/dx] * [u_(i,j)_(n) - u_(i-1,j)_(n)] - c * [dt/dy] * [u_(i,j)_(n) - u_(i,j-1)_(n)]
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	c = C				# wavespeed
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
								  (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
			
		U.append(u.copy())
	
	return np.array(U)
	
# STEP 2 - 1D NON-LINEAR CONVECTION	
def linear_convection_2D(X, Y, V, T, C):
	# du/dt + c * [du/dx] + c * [du/dy] = 0
	# solution: u_(i,j)_(n+1) = u_(i,j)_(n) - c * [dt/dx] * [u_(i,j)_(n) - u_(i-1,j)_(n)] - c * [dt/dy] * [u_(i,j)_(n) - u_(i,j-1)_(n)]
	
	nx = len(X)			# size of the x-space
	ny = len(Y)			# size of the y-space
	dx = X[1]-X[0]		# change in x-space
	dy = Y[1]-Y[0]		# change in y-space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	c = C				# wavespeed
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	for n in range(nt + 1): ##loop across number of time steps
		un = u.copy()
		u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -
								  (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
			
		U.append(u.copy())
	
	return np.array(U)
	

def velocity_function(X, Y):
	V = np.zeros([len(X), len(Y)])
	for i in np.arange(len(X)):
		for j in np.arange(len(Y)):
			V[i,j] = np.cos(np.abs(X[i])+np.abs(Y[j]))*(np.abs(X[i])+np.abs(Y[j]))
			
	return V


	
# EQUATION RUNNER
################################################################################################################

Interval = 15
C = 0.1
nu = 0.05
X = np.linspace(0, 2, 41)
Y = np.linspace(0, 2, 41)
T = np.linspace(0, 12, 81)
V = np.array(velocity_function(X,Y))

fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(111, projection='3d')                    
xx, yy = np.meshgrid(X, Y)
ness = linear_convection_2D(X, Y, V, T, C)

surf = None
def animate_2D(i):
	global surf
	if surf:
		ax.collections.remove(surf)
		
	surf = ax.plot_surface(xx, yy, ness[i], cmap=cm.viridis)

anim = animation.FuncAnimation(fig, animate_2D, frames=ness.shape[0], interval=Interval)
plt.show()
	