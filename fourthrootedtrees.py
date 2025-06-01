import sympy as sp

def rooted_tree_analysis_esrk_4th_order(s):
    assert s >= 8, "ESRK 2S−7 structure only valid for s ≥ 8"

    num_params = 2 * s - 7
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]

    # Step 1: b_i setup — p0 is shared across first 8 stages
    b_shared = params[0]
    b_unique = params[1 : 1 + (s - 8)]
    b_vars = [b_shared if i < 8 else b_unique[i - 8] for i in range(s)]

    # Step 2: a_{i,j} — only a_{i,i-1} are free; others tied to b_j
    a_vars = {}
    a_params = params[1 + (s - 8):]
    for i in range(s):
        for j in range(i):
            if j == i - 1:
                a_vars[(i, j)] = a_params[i]  # one free a per row
            else:
                a_vars[(i, j)] = b_vars[j]    # tied to b_j

    # Step 3: c_i = sum of a_{i,j}
    c_vars = {i: sum(a_vars[(i, j)] for j in range(i)) for i in range(s)}

    # Step 4: Rooted tree order conditions (orders 1–4)
    b, c, a = b_vars, c_vars, a_vars
    eqs = []
    eqs.append(sum(b) - 1)
    eqs.append(sum(b[i] * c[i] for i in range(s)) - sp.Rational(1, 2))
    eqs.append(sum(b[i] * c[i]**2 for i in range(s)) - sp.Rational(1, 3))
    eqs.append(sum(b[i] * sum(a[(i, j)] * c[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 6))
    eqs.append(sum(b[i] * c[i]**3 for i in range(s)) - sp.Rational(1, 4))
    eqs.append(sum(b[i] * sum(a[(i, j)] * c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1, 8))
    eqs.append(sum(b[i] * c[i] * sum(a[(i, j)] * c[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 12))
    eqs.append(sum(
        b[i] * sum(
            a[(i, j)] * sum(
                a[(j, k)] * c[k] for k in range(j)
            ) for j in range(i)
        ) for i in range(s)
    ) - sp.Rational(1, 24))

    return {
        "Stages (s)": s,
        "2s - 7 Parameters": num_params,
        "Rooted Tree Conditions": len(eqs),
        "Sufficient Parameters?": num_params >= len(eqs),
        "Structure": "Compressed ESRK (Van der Houwen style)",
        "Jacobian-Free": True  # symbolic only
    }

# Example usage
result = rooted_tree_analysis_esrk_4th_order(32)
for k, v in result.items():
    print(f"{k}: {v}")
