import scipy.io
import cvxpy as cp
import numpy as np


data = scipy.io.loadmat('hw06_data\lp_data.mat')


print(data.keys())
# print(data)
print("c", data["c"].shape)
n = data["c"].shape[0]

c = data["c"]
b = data["b"].squeeze()
A = data["A"]
G = data["G"]
h = data["h"].squeeze()
print("A:", A)
print(isinstance(A, np.ndarray))
print("b:", b)
print("c:", c)
print("G:", G)
print("h:", h)

x = cp.Variable(n)
print("(G).shape", (G).shape)
print("(G@x).shape", (G@x).shape)
print("h", h.shape)

objective = cp.Minimize(c.T@x)

constraints = [
  G@x <= h,
  A@x == b,
]


problem = cp.Problem(objective, constraints)
solution = problem.solve()

# Print results
print("Optimal value:", solution)
print("Optimal solution:", x.value)
print("Dual variables (inequalities):", problem.constraints[0].dual_value)

# Checking complimentary slackness.

slackness = h - G @ x.value
print("values of f_i(x^*) which are supposed to be <= 0:", slackness)

print("Note that complimentary slackness holds since all values are 0 (till precision of computers).",problem.constraints[0].dual_value * slackness)

# to check that x^* is indeed optimal, all we need to do is to check for conditions 0 and 1 from the big statement.

# Condition 0: Primal feasibility
print("Primal feasibility check equality constraints (should be close to zero):", (A @ x.value - b))
print("Primal feasibility check inequality constraints (should be close to zero or negative):", (G @ x.value - h))

# Hence we can see feasibility holds.

print("primal objective f(~x) = :", c.T @ x.value)
# print(f"shapes: h.T: {h.shape}, equality dual_value: {problem.constraints[1].dual_value.shape}, b.T: {b.T.shape}, inequality dual_value: {problem.constraints[0].dual_value.shape}")
print("dual objective g(~lambda, ~nu):",- (h.T @ problem.constraints[0].dual_value + b.T @ problem.constraints[1].dual_value))

print("these values are equal so strong duality holds and hence x^* is optimal.")