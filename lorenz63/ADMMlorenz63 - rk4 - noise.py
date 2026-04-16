#-----------------------------------------------------------------------
# This program compares linearized multi-block ADMM with L-BFGS-B and CG
# when recovering Lorenz 63 model with fourth-order Runge-Kutta method.
#-----------------------------------------------------------------------
# By Bowen Li
# October 7, 2024
#-----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from scipy.optimize import minimize
from scipy.integrate import odeint
import time


def g(x, t):

   """lorenz 63 model with fourth-order Runge-Kutta"""
   # Setting up vector
   d = np.zeros(N)

   d[0] = ss*(x[1] - x[0])
   d[1] = r*x[0] - x[1] - x[0]*x[2]
   d[2] = x[0]*x[1] - b*x[2]
   return d

# The Jacobi of g(x)
def dg(x):
   A = np.zeros((N, N))
   A[0, 0] = - ss
   A[0, 1] = ss
   A[1, 0] = r - x[2]
   A[1, 1] = -1
   A[1, 2] = -x[0]
   A[2, 0] = x[1]
   A[2, 1] = x[0]
   A[2, 2] = -b
   return A

# The iteration map x_{i} = f(x_{i-1})
def f(x, dt):
   t = 0
   k1 = g(x, t)
   k2 = g(x + 0.5*dt*k1, t)
   k3 = g(x + 0.5*dt*k2, t)
   k4 = g(x + dt*k3, t)
   return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# Tangent linear model
def df(x, dt):
   t = 0
   k1 = g(x, t)
   k2 = g(x + 0.5*dt*k1, t)
   k3 = g(x + 0.5*dt*k2, t)
   dk1 = dg(x)
   dk2 = np.matmul(dg(x + 0.5*dt*k1), np.eye(N) + 0.5*dt*dk1)
   dk3 = np.matmul(dg(x + 0.5*dt*k2), np.eye(N) + 0.5*dt*dk2)
   dk4 = np.matmul(dg(x + dt*k3), np.eye(N) + dt*dk3)
   return np.eye(N) + dt*(dk1 + 2*dk2 + 2*dk3 + dk4)/6


# The adjoint model
def f_adj(x, xt, dt):
   t = 0
   k1 = g(x, t)
   k2 = g(x + 0.5*dt*k1, t)
   k3 = g(x + 0.5*dt*k2, t)
   dk1 = dg(x)
   dk2 = np.matmul(dg(x + 0.5*dt*k1), np.eye(N) + 0.5*dt*dk1)
   dk3 = np.matmul(dg(x + 0.5*dt*k2), np.eye(N) + 0.5*dt*dk2)
   dk4 = np.matmul(dg(x + dt*k3), np.eye(N) + dt*dk3)
   A = dt*(dk1 + 2*dk2 + 2*dk3 + dk4)/6
   return xt + np.matmul(A.T, xt)

# Solve the subproblems
def subproblem(x0, x1, x2, lmda1, lmda2, xs, dt, rho, *, l=1, m=1, n=1):
   # The linearized subproblem
   # Use l,m,n to control which term appears in the subproblem
   # x1 is the variable to update

   eta = 0.1  # The parameter \eta for linearization in ADMM
   para = 1 + eta * (2*l + rho * m)
   b = 2*l*xs + m * (rho * f(x0, dt) + lmda1) + n * np.matmul(df(x1, dt).T, rho * x2 - rho*f(x1, dt) - lmda2)
   x1 = (x1 + eta * b)/para
   return x1

# Evaluate the objective function given the initial value
def obj(x0):
   fv = 0.5*alpha*np.linalg.norm(x0 - xs[0])**2
   xt = odeint(g, x0, t)
   for k in range(num_steps//M + 1):
      fv += (0.5/(num_steps//M)) * np.linalg.norm(xt[k*M] - xs[k*M])**2
   return fv

# The gradient of objective function with respect to initial value via adjoint method
def grad_cost(x):
    xstate=np.zeros([num_steps + 1, N], float)

    # ----- forward run -----
    x0=x.copy()
    for it in range(num_steps):
        xstate[it]=x0.copy()
        x0=f(x0, dt)
    xstate[num_steps]=x0
    # ----- end forward run -----

    # ----- adjoint run -----
    x0_ad=np.zeros(N)
    xout_ad=np.zeros(N)
    for it in range(num_steps, 0, -1):
        if it % M == 0:
            x0_ad += xstate[it]-xs[it]
        xout_ad=xout_ad+x0_ad
        x0_ad[()]=0.
        x0_ad=f_adj(xstate[it-1], xout_ad, dt)
        xout_ad[()]=0.
    # ----- end adjoint run -----
    return (0.5/(num_steps//M)) * x0_ad + alpha*(x - xs[0])



# These are our constants
N = 3  # Number of variables

# Constants for Lorenz63
ss=10
r=28
b=8/3

x0 = [-0.5, 0.5, 20.5]  # Initial state (equilibrium)

dt = 0.01  # The size of time steps
num_steps = 300  # The number of time steps
alpha = 0.1  # The parameter for background error (Here we set the background information equal to the inital observation for simplicity)


# The simulation of Lorenz 63 model
t = np.arange(0.0, (num_steps + 1)*dt, dt)

xst = odeint(g, x0, t)
print(xst[0])


# Add noise to the observation data
seed = 0
np.random.seed(seed)

xs = np.zeros((num_steps + 1, N))
for i in range(num_steps + 1):
   xs[i] = xst[i] + np.random.normal(0, 1, N)



iter_steps = 5000  # Number of iterations for ADMM
M = num_steps//10  # Take observations ever M steps (Totally 10 observations)
rho = 1.5  # The parameter \rho in ADMM
mu = 100  # The scaling parameter \mu in ADMM

x = np.zeros((num_steps + 1, N))  # The main variable, corresponding to the variable u in the paper
xt = np.zeros((num_steps + 1, N))  # Intermediate variable for x
lmda = np.zeros((num_steps + 1, N))  # The array for \lambda (The first element is not used)
lmdat = np.zeros((num_steps + 1, N))  # Intermediate variable for \lambda
fval = np.zeros(iter_steps + 1)  # Store the value of objective function for each iteration
cons_err = np.zeros(iter_steps + 1)  # Store the constraint error for each iteration

time_start = time.time()  # Record time


# Set the initial value for optimization methods
x[0] = [-3, -3, 10]
for k in range(num_steps):
   x[k+1] = f(x[k], dt)


# Save the value of objective function for initial value

#for k in range(num_steps//M + 1):
#   fval[0] += np.linalg.norm(x[k*M] - xs[k*M])**2
#fval[0] *= 0.5/(num_steps//M)
fval[0] += obj(x[0])

# Save the constraint error for initial value
for k in range(num_steps):
   cons_err[0] += np.linalg.norm(x[k+1] - f(x[k], dt))**2

# The multi-block ADMM with Jacobian decomposition
for i in range(iter_steps):
   xt[0] = subproblem(x[0], x[0], x[1], lmda[1], lmda[1], xs[0], dt, rho, l=(0.5/(num_steps//M))*mu + 0.5*alpha*mu, m=0)
   for j in range(1, num_steps):
      if j % M == 0:
         xt[j] = subproblem(x[j-1], x[j], x[j+1], lmda[j], lmda[j+1], xs[j], dt, rho, l=(0.5/(num_steps//M))*mu)
      else:
         xt[j] = subproblem(x[j-1], x[j], x[j+1], lmda[j], lmda[j+1], xs[j], dt, rho, l=0)
      # Update \lambda
      lmdat[j] = lmda[j] - rho*(xt[j] - f(xt[j - 1], dt))

   xt[num_steps] = subproblem(x[num_steps-1], x[num_steps], x[num_steps], lmda[num_steps], lmda[num_steps], xs[num_steps], dt, rho, l=(0.5/(num_steps//M))*mu, n=0)
   # Update \lambda
   lmdat[num_steps] = lmda[num_steps] - rho*(xt[num_steps] - f(xt[num_steps - 1], dt))
   
   # Update all the variables to implement Jacobi decomposed ADMM
   for j in range(num_steps + 1):
      x[j] = xt[j]
      lmda[j] = lmdat[j]

   # Store the value of objective function
   for k in range(num_steps//M + 1):
      fval[i + 1] += np.linalg.norm(x[k*M] - xs[k*M])**2
   fval[i + 1] *= 0.5/(num_steps//M)
   #fval[i + 1] += obj(x[0])

   # Store the constraint error
   for k in range(num_steps):
      cons_err[i + 1] += np.linalg.norm(x[k+1] - f(x[k], dt))**2


x_admm = odeint(g, x[0], t)

# Display the results
print("\n")
print("ADMM: ")
print(x[0])
print("Error:", fval[iter_steps])


print("Constraint Error:", cons_err[iter_steps])

time_end = time.time()
print("Time: "+str(time_end - time_start)+" Seconds")





# Adjoint method
#x0 = [0, 1, 5]  # this initial point is good for cg to converge, but not bfgs
x0 = [-3, -3, 10]  # this initial point is not good for both cg and bfgs, stuck in local minimum


fval_bfgs = []
fval_cg = []


fval_bfgs.append(obj(x0))
fval_cg.append(obj(x0))


def callback_bfgs(x0):
   fval_bfgs.append(obj(x0))
   return

def callback_cg(x0):
   fval_cg.append(obj(x0))
   return



# Optimization via L-BFGS-B
time_start_bfgs = time.time()
xout = minimize(obj, x0, method='L-BFGS-B', jac=grad_cost, tol=1e-12, options={'ftol': 0, 'gtol': 0, 'maxiter': iter_steps, 'maxfun': 1500000}, callback = callback_bfgs).x

x_bfgs = odeint(g, xout, t)

# Display the results
print("\n")
print("L-BFGS: ")
print(xout)
print("Error:", obj(xout))

time_end_bfgs = time.time()
print("Time: " + str(time_end_bfgs - time_start_bfgs) + " Seconds")


# Optimization via CG
time_start_cg = time.time()
xout = minimize(obj, x0, method='CG', jac=grad_cost, tol=0, options={'gtol': 0, 'maxiter': iter_steps}, callback = callback_cg).x

x_cg = odeint(g, xout, t)

print("\n")
print("CG: ")
print(xout)
print("Error:", obj(xout))

time_end_cg = time.time()
print("Time: " + str(time_end_cg - time_start_cg) + " Seconds")






# Convergence graph of function value
fig = plt.figure(0, dpi=60, figsize=(15, 13))
ax = fig.add_subplot(111)
ax.plot(range(iter_steps + 1), fval, label='ADMM', color='#ff7f0e', linewidth=2)
#ax.plot(range(len(fval_bfgs)), fval_bfgs, label='L-BFGS', color='#d62728', linewidth=2)
#ax.plot(range(len(fval_cg)), fval_cg, label='CG', color='#2ca02c', linewidth=2)

# #1f77b4 Blue; #d62728 Red; #ff7f0e Orange; #2ca02c Green

ax.axvline(x=1000, color='k', linestyle='--')

ax.set_xlabel("Iteration", fontsize=40, labelpad=15)
ax.set_ylabel(r"$F - F^\star$", fontsize=40, labelpad=15)

# Set the size of the tick mark
ax.tick_params(axis='both', labelsize=32)

ax.legend(fontsize=37, loc='upper right', frameon=True, framealpha=0.8, edgecolor='black')

# Set the logarithmic ordinate
ax.set_yscale('log')

plt.savefig('convergence.pdf', dpi=60, format='pdf', bbox_inches='tight')  # Save a high-quality image

#plt.show()






# Convergence graph of constraint error
fig = plt.figure(1, dpi=60, figsize=(15, 13))
ax = fig.add_subplot(111)
ax.plot(range(1, iter_steps + 1), cons_err[1:], label='ADMM', color='#ff7f0e', linewidth=2)
#ax.plot(range(len(fval_bfgs)), fval_bfgs, label='L-BFGS', color='#d62728', linewidth=2)
#ax.plot(range(len(fval_cg)), fval_cg, label='CG', color='#2ca02c', linewidth=2)

# #1f77b4 Blue; #d62728 Red; #ff7f0e Orange; #2ca02c Green

ax.axvline(x=1000, color='k', linestyle='--')

ax.set_xlabel("Iteration", fontsize=40, labelpad=15)
ax.set_ylabel(r"$\sum_{k=0}^{N-1}\ \left\| \mathbf{u}_{k+1}-H(\mathbf{u}_k) \right\|^2$", fontsize=32, labelpad=15)

# Set the size of the tick mark
ax.tick_params(axis='both', labelsize=32)

ax.legend(fontsize=37, loc='upper right', frameon=True, framealpha=0.8, edgecolor='black')

# Set the logarithmic ordinate
ax.set_yscale('log')

plt.savefig('cons_err.pdf', dpi=60, format='pdf', bbox_inches='tight')  # Save a high-quality image

#plt.show()





# Draw the dynamics
fig = plt.figure(2, dpi=60, figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs[::M,0], xs[::M,1], xs[::M,2], label='Obs', c='magenta', marker='o', s=100, alpha=0.7)
ax.plot(xst[:,0], xst[:,1], xst[:,2], label='True', color='#1f77b4', linewidth=2)

ax.plot(x_admm[:,0], x_admm[:,1], x_admm[:,2], label='ADMM', color='#ff7f0e', linestyle='-.', linewidth=2)
#ax.plot(x_bfgs[:,0], x_bfgs[:,1], x_bfgs[:,2], label='L-BFGS-B', color='#d62728', linewidth=2)
#ax.plot(x_cg[:,0], x_cg[:,1], x_cg[:,2], label='CG-PR', color='#2ca02c', linewidth=2)

# Adjust the perspective
ax.view_init(elev=30, azim=320)  # elev is the elevation angle, and azim is the azimuth angle

ax.legend(fontsize=37, loc='upper right', frameon=True, framealpha=0.8, edgecolor='black', bbox_to_anchor=(1.075, 0.95), borderaxespad=0)

# Adjust the distance of the X, Y, and Z axis tick labels
ax.xaxis.set_tick_params(pad=6)  # Adjust the distance between the X-axis tick labels and the axis
ax.yaxis.set_tick_params(pad=6)  # Adjust the distance between the Y-axis tick labels and the axis
ax.zaxis.set_tick_params(pad=6)  # Adjust the distance between the Z-axis tick labels and the axis

ax.set_xlabel('X', fontsize=37, labelpad=25)
ax.set_ylabel('Y', fontsize=37, labelpad=25)
ax.set_zlabel('Z', fontsize=37, labelpad=25)

ax.tick_params(axis='both', which='major', labelsize=32)
ax.tick_params(axis='z', labelsize=32)

ax.set_xticks(np.linspace(-20, 10, 4))  # Set the X-axis range
ax.set_yticks(np.linspace(-10, 20, 4))  # Set the Y-axis range
ax.set_zticks(np.linspace(10, 40, 4))  # Set the Z-axis range

# Set the background to white
ax.set_facecolor('white')

# Remove the grid lines
ax.grid(False)

plt.savefig('dynamic.pdf', dpi=60, format='pdf', bbox_inches='tight')  # Use high DPI when saving

plt.show()
