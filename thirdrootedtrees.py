import sympy as sp
import numpy as np

def verify_2s_minus_8_matches_rooted_trees(s, nprobe=5):
    """
    Fast numeric-rank test for the 2S-8 low-storage RK template satisfying 3rd-order conditions.
    """

    assert s >= 9, "Need at least 9 stages for 2S-8"

    # Total parameters for 2S-8
    num_params = 2*s - 8
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]

    # 1) b vars: tie b[0..7], free b[8..s-1]
    b_shared = params[0]
    b_unique = params[1 : 1 + (s - 8)]  # length = s-8
    b_vars = [b_shared if i < 8 else b_unique[i - 8] for i in range(s)]

    # 2) a vars: free a_{i,i-1}, rest tied to b_j
    a_vars = {}
    a_start = 1 + (s - 8)
    a_params = params[a_start:]  # length = s-1
    for i in range(s):
        for j in range(i):
            if j == i - 1:
                a_vars[(i, j)] = a_params[i - 1]
            else:
                a_vars[(i, j)] = b_vars[j]

    # 3) c values
    c_vars = {i: sum(a_vars[(i, j)] for j in range(i)) for i in range(s)}

    # 4) third-order conditions (4 eqs)
    eqs = [
        sum(b_vars) - 1,  # order-1
        sum(b_vars[i]*c_vars[i] for i in range(s)) - sp.Rational(1, 2),  # order-2
        sum(b_vars[i]*c_vars[i]**2 for i in range(s)) - sp.Rational(1, 3),  # order-3 (diagonal part)
        sum(b_vars[i]*sum(a_vars[(i, j)]*c_vars[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 6)  # order-3 (off-diagonal)
    ]

    # 5) Build symbolic Jacobian and lambdify
    J_sym = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in eqs])
    J_fun = sp.lambdify(params, J_sym, "numpy")

    # 6) Numeric rank tests
    best_rank = 0
    for _ in range(nprobe):
        vals = np.random.rand(num_params)
        J_num = np.array(J_fun(*vals), dtype=float)
        rank = np.linalg.matrix_rank(J_num)
        best_rank = max(best_rank, rank)
        if best_rank == len(eqs):
            break

    return {
        "Stages (s)": s,
        "2s - 8 Parameters": num_params,
        "Order Conditions": len(eqs),
        "Max Numeric Rank": best_rank,
        "Matches 2s - 8": best_rank == len(eqs)
    }

# Example usage
if __name__ == "__main__":
    for s in [15,20,25,30,35,40,45,50,55,65,70]:
        result = verify_2s_minus_8_matches_rooted_trees(s, nprobe=10)
        print(result)
