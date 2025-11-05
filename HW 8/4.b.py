import scipy.io
import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt



data = scipy.io.loadmat('data\ellipsoid_data.mat')

print(data.keys())
N = data['N']
M = data['M']
X = data['X']
Y = data['Y']

n = 2 # dimension of the space

print(f"N.shape: {N.shape}")
print(f"M.shape: {M.shape}")
print(f"X.shape: {X.shape}")
print(f"Y.shape: {Y.shape}")

print(N)
print(M)


P = cp.Variable((n, n), symmetric=True)
q = cp.Variable(n)
r = cp.Variable()
gamma = cp.Variable()

constraints = []

for i in range(N[0][0]):
    constraints.append(cp.quad_form(X[:, i], P) + q.T @ X[:, i] + r <= 0)

for i in range(M[0][0]):
    constraints.append(cp.quad_form(Y[:, i], P) + q.T @ Y[:, i] + r >= 0)

identity_matrix = np.eye(n)
constraints.append(P - identity_matrix >> 0)

constraints.append(gamma * identity_matrix - P >> 0)

objective = cp.Minimize(gamma)

problem = cp.Problem(objective, constraints)


problem.solve()

if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
    print(f"Problem status: {problem.status}")
    print(f"Optimal value of gamma: {gamma.value}")
    print("Optimal matrix P:\n", P.value)
    print("Optimal vector q:\n", q.value)
    print(f"Optimal scalar r: {r.value}")
else:
    print(f"Problem could not be solved. Status: {problem.status}")



# Creating the boundary of the ellipsoid for visualization

A = np.linalg.inv(P.value)
x_0 = -1/2 * A @ q.value
d = np.sqrt(x_0.T @ P.value @ x_0 - r.value)

theta = np.linspace(0, 2 * np.pi, 1000)
ellipse_x = d * np.cos(theta)
ellipse_y = d * np.sin(theta)

all_points = np.vstack((ellipse_x, ellipse_y))
elipse_boundary_points = A@all_points+ x_0.reshape(-1,1)




plt.figure(figsize=(8, 6))


plt.scatter(elipse_boundary_points[0, :], elipse_boundary_points[1, :], c='black', label='Boundary of Ellipsoid', s=1)


plt.scatter(X[0, :], X[1, :], c='blue', label='Dataset X')


plt.scatter(Y[0, :], Y[1, :], c='red', label='Dataset Y')

plt.title('Scatter Plot of Datasets X and Y')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid(True)
plt.axis('equal') # Ensures the scaling is the same on both axes
plt.show()