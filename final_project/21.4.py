import cvxpy as cp
import numpy as np

np.random.seed(0)
n = 100   #number of ads
m = 30    #number of contracts
T = 60    #number of periods

#number of impressions in each period
I = 10*np.random.rand(T)
#revenue rate for each period and ad
R = np.random.rand(n, T)
#contract target number of impressions
q = T/float(n)*50*np.random.rand(m)
#penalty rate for shortfall
p = np.random.rand(m)
#one column per contract. 1's at the periods to be displayed
Tcontr = np.matrix(np.random.rand(T, m)>.8, dtype = float)
Acontr = np.zeros((n, m))
Acont = Acontr
for i in range(n):
    contract=int(np.floor(m*np.random.rand(1)))
    #one column per contract. 1's at the ads to be displayed
    Acontr[i,contract]=1

Tcontr = np.array(Tcontr)
Acontr = np.array(Acontr)

N_greedy = np.zeros((n, T)) # Greedy solution
revenue_greedy = 0

for t in range(T):
    best_ad = np.argmax(R[:, t]) # Just choose the ad that gives the highest revenue in each time period.
    N_greedy[best_ad, t] = I[t]
    revenue_greedy += R[best_ad, t] * I[t]

penalty_greedy = 0
for j in range(m):
    relevant_ads = np.where(Acontr[:, j] == 1)[0]
    relevant_periods = np.where(Tcontr[:, j] == 1)[0]

    served = 0
    if len(relevant_ads) > 0 and len(relevant_periods) > 0:
        served = np.sum(N_greedy[np.ix_(relevant_ads, relevant_periods)])

    shortfall = max(0, q[j] - served)
    penalty_greedy += p[j] * shortfall

print(f"Greedy Net Profit: {revenue_greedy - penalty_greedy:.2f}")
print(f"Greedy Revenue: {revenue_greedy:.2f}")
print(f"Greedy Penalty: {penalty_greedy:.2f}")


N = cp.Variable((n, T), nonneg=True)

total_revenue = cp.sum(cp.multiply(R, N)) # Total revenue


penalty_terms = []
for j in range(m):
    relevant_ads = np.where(Acontr[:, j] == 1)[0]
    relevant_periods = np.where(Tcontr[:, j] == 1)[0]

    served_expr = 0
    if len(relevant_ads) > 0 and len(relevant_periods) > 0:
        served_expr = cp.sum(N[np.ix_(relevant_ads, relevant_periods)])

    penalty_terms.append(p[j] * cp.pos(q[j] - served_expr)) # cp.pos(x) gives max(0, x)

total_penalty = cp.sum(penalty_terms)

constraints = [cp.sum(N[:, t]) == I[t] for t in range(T)]


prob = cp.Problem(cp.Maximize(total_revenue - total_penalty), constraints) # Actual problem
prob.solve()

print(f"Optimal Net Profit: {prob.value:.2f}")
print(f"Optimal Revenue: {total_revenue.value:.2f}")
print(f"Optimal Penalty: {total_penalty.value:.2f}")