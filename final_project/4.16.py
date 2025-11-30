import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Problem Data
A = np.array([[-1, 0.4, 0.8],
              [1, 0, 0],
              [0, 1, 0]])
b = np.array([1, 0, 0.3])
x_des = np.array([7, 2, -6])
N = 30
n = 3

# Variables
u = cp.Variable(N) # Main decision variable
x = cp.Variable((n, N+1)) # state of the machine


fuel_usage = cp.sum(cp.maximum(cp.abs(u), 2*cp.abs(u) - 1))# To minimize this

# Constraints
constraints = [x[:, 0] == np.zeros(n)] # make sure it starts at zero
for t in range(N):
    constraints.append(x[:, t+1] == A @ x[:, t] + b * u[t]) # it needs to follow the dynamics
    # x(t+1) = A*x(t) + b*u(t)
constraints.append(x[:, N] == x_des) # make sure that it reaches the x_des.

# Solve
problem = cp.Problem(cp.Minimize(fuel_usage), constraints)
solution = problem.solve()

print(f"Optimal fuel consumption: {solution:.4f}")
print(u.value)

u_value = u.value
t_steps = np.arange(N)

plt.figure(figsize=(10, 5))
plt.plot(t_steps, u_value, label=r'$u$')

plt.xlabel(r'Time $t$', fontsize=12)
plt.ylabel(r'Input $u(t)$', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5) # Zero line
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()