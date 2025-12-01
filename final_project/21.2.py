# radiation treatment planning data file
# this one is randomly generated; a real one would use the beam geometry
import scipy.io as sio
import numpy as np
import cvxpy as cp

def get_value(mat, name):
    value = mat[name]
    return value[0][0] if np.shape(value) == (1, 1) else value

mat = sio.loadmat("./treatment_planning_data.mat")

n = get_value(mat, "n")  # number of beams
mtarget = get_value(mat, "mtarget")  # number of tumor or target voxels
mother = get_value(mat, "mother")  # number of other voxels
Atarget = get_value(mat, "Atarget")
Aother = get_value(mat, "Aother")
Bmax = get_value(mat, "Bmax")
Dtarget = get_value(mat, "Dtarget")
Dother = get_value(mat, "Dother")


# Until here, the code was provided on github.

b = cp.Variable(n) # beam vector that we can control

objective = cp.Minimize(cp.sum_squares(cp.pos(Aother @ b - Dother))) # Objective set up the same way as specified

constraints = [
    Atarget @ b >= Dtarget,
    b >= 0,
    b <= Bmax
]

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal Objective Value:", problem.value)
print(b.value)