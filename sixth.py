import sympy as sp
import numpy as np
from scipy.optimize import approx_fprime


# Redefine for 2S-3 structure (share b[0..3])
def build_sixth_order_16_conditions_2s3(s, params):
    b_shared = params[0]
    b_unique = params[1:1 + (s - 4)]  # share b[0..3]
    b_vars = [b_shared if i < 4 else b_unique[i - 4] for i in range(s)]

    a_params = params[1 + (s - 4):]
    a_vars = {}
    for i in range(s):
        for j in range(i):
            if j == i - 1:
                a_vars[(i, j)] = a_params[i - 1]
            elif (i == s - 1) and (j == s - 3):
                a_vars[(i, j)] = a_params[s - 1]
            else:
                a_vars[(i, j)] = b_vars[j]

    c_vars = {i: sum(a_vars[(i, j)] for j in range(i)) for i in range(s)}
    b, c, a = b_vars, c_vars, a_vars

    eqs = []

    # 16 sixth-order conditions
    eqs.extend([
        sum(b[i] * c[i]**5 for i in range(s)) - sp.Rational(1, 6),
        sum(b[i] * sum(a[(i, j)] * c[j]**4 for j in range(i)) for i in range(s)) - sp.Rational(1, 15),
        sum(b[i] * c[i]**3 * sum(a[(i, j)] * c[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 30),
        sum(b[i] * c[i]**2 * sum(a[(i, j)] * c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1, 36),
        sum(b[i] * c[i] * sum(a[(i, j)] * c[j]**3 for j in range(i)) for i in range(s)) - sp.Rational(1, 45),
        sum(b[i] * sum(a[(i, j)] * sum(a[(j, k)] * c[k]**3 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 60),
        sum(b[i] * c[i] * sum(a[(i, j)] * sum(a[(j, k)] * c[k]**2 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 60),
        sum(b[i] * c[i]**2 * sum(a[(i, j)] * sum(a[(j, k)] * c[k] for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 60),
        sum(b[i] * sum(a[(i, j)] * c[j]**2 * sum(a[(j, k)] * c[k] for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 60),
        sum(b[i] * sum(a[(i, j)] * sum(a[(j, k)] * c[k] * sum(a[(k, l)] * c[l] for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 90),
        sum(b[i] * c[i] * sum(a[(i, j)] * c[j] * sum(a[(j, k)] * c[k] for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 90),
        sum(b[i] * sum(a[(i, j)] * c[j] * sum(a[(j, k)] * c[k]**2 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 90),
        sum(b[i] * c[i]**2 * sum(a[(i, j)] * c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1, 60),
        sum(b[i] * sum(a[(i, j)] * sum(a[(j, k)] * sum(a[(k, l)] * c[l]**2 for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 180),
        sum(b[i] * sum(a[(i, j)] * sum(a[(j, k)] * c[k] * sum(a[(k, l)] * c[l] for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 180),
        sum(b[i] * sum(a[(i, j)] * c[j] * sum(a[(j, k)] * sum(a[(k, l)] * c[l] for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1, 180)
    ])
    return eqs

def verify_sixth_order_2s3_trimmed(s, nprobe=5, epsilon=1e-6):
    num_params = 2 * s - 3
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]
    eqs = build_sixth_order_16_conditions_2s3(s, params)
    eq_func = sp.lambdify(params, eqs, "numpy")

    best_rank = 0
    for trial in range(nprobe):
        x0 = np.random.rand(num_params)
        J = np.vstack([approx_fprime(x0, lambda x: eq_func(*x)[i], epsilon) for i in range(len(eqs))])
        U, Svals, Vt = np.linalg.svd(J, full_matrices=False)
        tol = max(J.shape) * np.finfo(float).eps * Svals.max()
        rank = int((Svals > tol).sum())
        best_rank = max(best_rank, rank)
        print(f"  Trial {trial:2d}: 2S-3 rank = {rank} / 16")
        if rank == 16:
            break

    return {
        "Stages (s)": s,
        "2s - 3 Parameters": num_params,
        "Sixth-Order Conditions (Trimmed)": 16,
        "Rank Achieved": best_rank,
        "Full Rank?": (best_rank == 16),
        "Structure": "2S−3 ESRK (16 Cond. Sixth Order)"
    }

# Try for s = 22
result_2s3 = verify_sixth_order_2s3_trimmed(s=25)

import pandas as pd
import ace_tools as tools
tools.display_dataframe_to_user(name="Sixth-Order 2S-3 Trimmed", dataframe=pd.DataFrame([result_2s3]))
