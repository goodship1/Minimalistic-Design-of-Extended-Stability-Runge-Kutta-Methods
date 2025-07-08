import sympy as sp
import numpy as np

def verify_2s_minus_8_fast(s, nprobe=5):
    assert s >= 9, "Need at least 9 stages for 2S-8"
    num_params = 2*s - 8
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]

    # 1) b vars: tie b[0..7], free b[8..s-1]
    b_shared = params[0]
    b_unique = params[1:1 + (s - 8)]
    b = [b_shared if i < 8 else b_unique[i - 8] for i in range(s)]

    # 2) a vars: free a_{i,i-1}, rest = b_j
    a_start = 1 + (s - 8)
    a_params = params[a_start:]  # length = s-1
    a = {}
    for i in range(s):
        for j in range(i):
            a[i,j] = a_params[i - 1] if j == i-1 else b[j]

    # 3) c values
    c = {i: sum(a[i,j] for j in range(i)) for i in range(s)}

    # 4) third-order conditions (4 eqs)
    eqs = [
        sum(b) - 1,
        sum(b[i]*c[i] for i in range(s)) - sp.Rational(1,2),
        sum(b[i]*c[i]**2 for i in range(s)) - sp.Rational(1,3),
        sum(b[i]*sum(a[i,j]*c[j] for j in range(i)) for i in range(s)) - sp.Rational(1,6)
    ]

    # 5) symbolic Jacobian → numeric function
    J_sym = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in eqs])
    J_fun = sp.lambdify(params, J_sym, "numpy")

    # 6) numeric rank tests
    best_rank = 0
    for _ in range(nprobe):
        vals = np.random.rand(num_params)
        Jn = np.array(J_fun(*vals), float)
        best_rank = max(best_rank, np.linalg.matrix_rank(Jn))
        if best_rank == len(eqs):
            break

    return {
        "Stages (s)": s,
        "2s - 8 Params": num_params,
        "Order Conds": len(eqs),
        "Max Rank": best_rank,
        "Matches 2s-8": best_rank == len(eqs)
    }

if __name__ == "__main__":
    for s in [15,20,25,30,35,40,45,50,55]:
        print(verify_2s_minus_8_fast(s, nprobe=10))
