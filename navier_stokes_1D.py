import numpy as np
import sympy
from sympy import lambdify
import matplotlib.pyplot as plt
from matplotlib import animation

# animation function.  This is called sequentially
def animate_1D(i, X, U):
    line.set_data(X, U[i])
    return line,


# ONE DIMENSIONAL CFD EQUATIONS
# discretize with Forward Difference in time
# discretize with Backward Difference in space
	
# u is velocity
# t is time
# x is position
# c is wavespeed
# nu is viscosity
	
# STEP 1 - 1D LINEAR CONVECTION	
def linear_convection_1D(X, V, T, C):
	# du/dt + c * [du/dx] = 0
	# solution: u_(i)_(n+1) = u_(i)_(n) - c * [dt/dx] * [u_(i)_(n) - u_(i-1)_(n)]
	
	nx = len(X)			# size of the space
	dx = X[1]-X[0]		# change in space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	c = C
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	un = np.ones(nx)
	for n in range(nt):
		un = u.copy()
		for i in range(1, nx):
			u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
			
		U.append(u.copy())
	
	return X, U


# STEP 2 - 1D NONLINEAR CONVECTION
def nonlinear_convection_1D(X, V, T):
	# du/dt + f(u) * [du/dx] = 0
	# solution: u_(i)_(n+1) = u_(i)_(n) - f(u_(i)_(n)) * [dt/dx] * [u_(i)_(n) - u_(i-1)_(n)] 
	# f(u_(i)_(n)) = u_(i)_(n)
	
	nx = len(X)			# size of the space
	dx = X[1]-X[0]		# change in space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	un = np.ones(nx)
	for n in range(nt):
		un = u.copy()
		for i in range(1, nx):
			u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i-1])
			
		U.append(u.copy())
	
	return X, U

	
# STEP 3 - 1D DIFFUSION
def diffusion_1D(X, V, T, nu):
	# du/dt = nu * [(d^2)u/(dx^2)] = 0
	# solution: u_(i)_(n+1) = u_(i)_(n) + [nu*dt/[dx^2]] * [u_(i+1)_(n) - 2*u_(i)_(n) + u_(i-1)_(n)] 
	
	nx = len(X)			# size of the space
	dx = X[1]-X[0]		# change in space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	un = np.ones(nx)
	for n in range(nt):
		un = u.copy()
		for i in range(1, nx-1):
			u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])
			
		U.append(u.copy())
	
	return X, U

	
# STEP 4 - BURGERS' EQUATION
def burgers_equation_1D(X, V, T, nu):
	# du/dt + u * [du/dx] = nu * [(d^2)u/(dx^2)] = 0
	# solution: u_(i)_(n+1) = u_(i)_(n) - u_(i)_(n) * [dt/dx] * [u_(i)_(n) - u_(i-1)_(n)] + [nu*dt/[dx^2]] * [u_(i+1)_(n) - 2*u_(i)_(n) + u_(i-1)_(n)] 
	
	nx = len(X)			# size of the space
	dx = X[1]-X[0]		# change in space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	un = np.ones(nx)
	for n in range(nt):
		un = u.copy()
		for i in range(1, nx-1):
			u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                (un[i+1] - 2 * un[i] + un[i-1])
			
		U.append(u.copy())
	
	return X, U
	
def special_defn_burgers_equation_1D():
	# du/dt + u * [du/dx] = nu * [(d^2)u/(dx^2)] = 0
	# solution: u_(i)_(n+1) = u_(i)_(n) - u_(i)_(n) * [dt/dx] * [u_(i)_(n) - u_(i-1)_(n)] + [nu*dt/[dx^2]] * [u_(i+1)_(n) - 2*u_(i)_(n) + u_(i-1)_(n)] 
	
	# Establish Initial Conditions for Burgers' Equation:
	# u = [2*nu/phi] * [d/dx phi] + 4
	# phi = exp(-[x-4t]^2/[4*nu*[t-1]])+exp(-[x-4t-2*pi]^2/[4*nu*[t-1]])
	x, nu, t = sympy.symbols('x nu t')
	phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
		   sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
	phiprime = phi.diff(x)
	u = -2 * nu * (phiprime / phi) + 4
	ufunc = lambdify((t, x, nu), u)
	
	nu = 0.03
	X = np.linspace(0, 2*np.pi, 101)
	T = np.linspace(0, 6, 501)
	V = np.asarray([ufunc(0, x0, nu) for x0 in X])
	
	nx = len(X)			# size of the space
	dx = X[1]-X[0]		# change in space
	nt = len(T)			# number of time steps
	dt = T[1]-T[0]		# change in time
	
	# Starting velocity
	u = V.copy()

	# Loop over time and space
	U = []
	un = np.ones(nx)
	for n in range(nt):
		un = u.copy()
		for i in range(1, nx-1):
			u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                (un[i+1] - 2 * un[i] + un[i-1])
				
		# Establish Periodic Boundary Conditions:
		u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
                (un[1] - 2 * un[0] + un[-2])
		u[-1] = u[0]
			
		U.append(u.copy())
	
	return X, U
	

if __name__ == "__main__":
	
	# EQUATION RUNNER
	################################################################################################################
	
	Interval = 30
	nu = 0.05
	C = 0.1
	X = np.linspace(0, 8, 161)
	T = np.linspace(0, 8, 501)
	V = np.sin(np.pi*X)/10 - np.cos(0.5*np.pi*X)/10 + 1
	
	#onett, ness = linear_convection_1D(X, V, T, C)
	#onett, ness = nonlinear_convection_1D(X, V, T)
	#onett, ness = diffusion_1D(X, V, T, nu)
	onett, ness = burgers_equation_1D(X, V, T, nu)
	#onett, ness = special_defn_burgers_equation_1D()
	
	fig = plt.figure()
	ax = plt.axes(xlim=(min(onett)-1, max(onett)+1), ylim=(min(ness[0])-1, max(ness[0])+1))
	line, = ax.plot([], [], 'r-')
	
	anim = animation.FuncAnimation(fig, animate_1D, fargs=(onett, ness), frames=len(ness), interval=Interval, blit=True)
	plt.show()
	
	################################################################################################################
	