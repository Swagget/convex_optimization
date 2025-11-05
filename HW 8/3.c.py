import cvxpy as cp
import numpy as np

H1 = np.array([[1,0],[0,1],[-1,0],[0,-1]])
q1 = np.array([1,1,0,0])
# C is the set that satisfies  H1*x <= q1, where x is of the form [x_1,x_2,x_1,x_2]

H2 = np.array([[-1,0],[0,-1],[1,0],[0,1]])
q2 = np.array([2,2,-1,-1])
# D is the set that satisfies  H2*x <= q2, where x is of the form [x_1,x_2,x_1,x_2]

a = cp.Variable((2,1)) #I'll add the constraint for a_1=a_3 and a_2=a_4
lambda_1 = cp.Variable((4,1),nonneg=True)
lambda_2 = cp.Variable((4,1),nonneg=True)

objective = cp.Maximize(-q1.T@lambda_1-q2.T@lambda_2)

constraints_a = [cp.norm(a,2) <= 1,
               H1.T@lambda_1 + a == 0,
               H2.T@lambda_2 - a == 0]

problem = cp.Problem(objective, constraints_a)

solution_a = problem.solve()

print("Optimal value of a:", a.value)
# print("Optimal value of lambda_1:", lambda_1.value)
# print("Optimal value of lambda_2:", lambda_2.value)
# print("Optimal value of the objective function:", solution)

fixed_a = np.array(a.value)

closest_from_C = cp.Variable((2,1))

constraints_c = [H1@closest_from_C <= q1]

problem_c = cp.Problem(cp.Minimize(fixed_a.T@closest_from_C), constraints_c)

solution_c = problem_c.solve()

print("Optimal value of closest_from_C:", closest_from_C.value)


fixed_a_d = np.array(a.value)
print("fixed_a_d:",fixed_a_d)

closest_from_D = cp.Variable((2,1))
q2 = q2.reshape(-1,1)
constraints_d = [H2@closest_from_D <= q2]
print("H1:",H1, H1.shape)
print("H2:",H2, H2.shape)
print("q2:",q2,q2.shape)
print("q1:",q1,q1.shape)

problem_d = cp.Problem(cp.Maximize(fixed_a_d.T@closest_from_D), constraints_d)

solution_d = problem_d.solve()

print("Optimal value of closest_from_D:", closest_from_D.value)