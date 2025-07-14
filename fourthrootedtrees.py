import sympy as sp

def rooted_tree_analysis_esrk_4th_order(s):
    assert s >= 8, "ESRK 2S-8 structure only valid for s ≥ 8"

    # 1) Total number of parameters for 2S-8
    num_params = 2 * s - 8
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]

    # 2) b_i setup — tie b[0..7], free b[8..s-1]
    b_shared = params[0]
    b_unique = params[1 : 1 + (s - 8)]       # length = s-8
    b_vars   = [b_shared if i < 8 else b_unique[i - 8] for i in range(s)]

    # 3) a_{i,j} — free only for sub-diagonal i>=1: a_{i,i-1}
    a_start  = 1 + (s - 8)
    a_params = params[a_start : a_start + (s - 1)]  # exactly s-1 entries

    a_vars = {}
    for i in range(1, s):
        for j in range(i):
            if j == i - 1:
                # sub-diagonal free parameter for row i
                a_vars[(i, j)] = a_params[i - 1]
            else:
                # all other j<i tie back to b_j
                a_vars[(i, j)] = b_vars[j]

    # (a_{0,j} does not exist since j<i implies no j<0)

    # 4) c_i = sum_{j<i} a_{i,j}
    c_vars = {i: sum(a_vars[(i, j)] for j in range(i)) for i in range(s)}

    # 5) Rooted‐tree order conditions up to 4th order (8 equations)
    b, c, a = b_vars, c_vars, a_vars
    eqs = [
        sum(b) - 1,                                                       # order 1
        sum(b[i]*c[i] for i in range(s)) - sp.Rational(1, 2),            # order 2
        sum(b[i]*c[i]**2 for i in range(s)) - sp.Rational(1, 3),         # order 3 diagonal
        sum(b[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 6),  # order 3 off‐diag
        sum(b[i]*c[i]**3 for i in range(s)) - sp.Rational(1, 4),         # order 4
        sum(b[i]*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1, 12), # A(c^2)
        sum(b[i]*c[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 8),  # c·(A c)
        sum(
            b[i] * sum(
                a[(i,j)]
                * sum(a[(j,k)]*c[k] for k in range(j))
            for j in range(i))
        for i in range(s)
        ) - sp.Rational(1, 24)                                            # A(A c)
    ]

    return {
        "Stages (s)": s,
        "2S - 8 Parameters": num_params,
        "Rooted Tree Conditions": len(eqs),
        "Sufficient Parameters?": num_params >= len(eqs),
        "Structure": "2S-8 ESRK (Van der Houwen style)",
    }

# Test across many stage counts
if __name__ == "__main__":
    for stages in range(16, 300):
        out = rooted_tree_analysis_esrk_4th_order(stages)
        print(out)
