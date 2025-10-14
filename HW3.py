import cvxpy as cp
import numpy as np


print("Problem part a:")

x = cp.Variable(2)

cp.objective = cp.Minimize(np.matmul(np.ones(2).T, x))

constraint = [x >= np.array([1,-2])]

prob = cp.Problem(cp.objective, constraint)

x_final = prob.solve()

actual_vector = x.value

print(f"Value of x: {x_final}")
print(f"new x: {actual_vector}")





print("Problem part b:")

X = cp.Variable((2,2), symmetric=True)

cp.objective = cp.Minimize(cp.norm(X, 2))

constraint_matrix = np.array([[1,-2],[-2,1]])
constraint = [(X-constraint_matrix) >> 0]

problem = cp.Problem(cp.objective, constraint)
X_value = problem.solve()

new_X = X.value

print(f"Norm of X: {X_value}")
print(f"new X: {new_X}")
