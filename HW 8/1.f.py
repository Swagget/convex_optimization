import cvxpy as cp

from cvxpy.transforms import partial_optimize

x = cp.Variable()
unknown_lambda_first = cp.Variable(nonneg=True)
unknown_lambda_second = cp.Variable(nonneg=True)

objective_first = cp.Minimize(-cp.sqrt(x - unknown_lambda_first) - 0.5 * unknown_lambda_first)
objective_second = cp.Minimize(-cp.sqrt(3 - (x + unknown_lambda_second)) - 0.5 * unknown_lambda_second)

constraints_first = [x - unknown_lambda_first >= 0]
constraints_second = [x - unknown_lambda_second >= 0]

problem_first = cp.Problem(objective_first, constraints_first)
problem_second = cp.Problem(objective_second, constraints_second)

f_x = partial_optimize.partial_optimize(problem_first, opt_vars = [unknown_lambda_first])
f_3_minus_x = partial_optimize.partial_optimize(problem_second, opt_vars = [unknown_lambda_second])


full_problem = cp.Problem(cp.Minimize(f_x+f_3_minus_x), [x >= 0, x <= 3])
solution= full_problem.solve()

print("Optimal value of x:", x.value)
print("Optimal value of the objective function:", solution)
