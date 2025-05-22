import sympy as sp

def verify_2s_minus_7_matches_rooted_trees(s):
    assert s >= 8, "S must be at least 8 for 2S − 7 structure to hold"

    num_params = 2 * s - 7
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]

    # ---------------------
    # STEP 1: Assign b_i
    # ---------------------
    b_vars = []
    b_shared = params[0]
    b_unique = params[1 : 1 + (s - 8)]  # params[1] to params[s - 8]

    for i in range(s):
        if i < 8:
            b_vars.append(b_shared)
        else:
            b_vars.append(b_unique[i - 8])

    # ---------------------
    # STEP 2: Assign a_{i,j}
    # Only a_{i,i-1} are free; rest tied to b_j
    # ---------------------
    a_vars = {}
    a_start = 1 + (s - 8)
    a_params = params[a_start:]  # these are the free a_{i,i−1}

    if len(a_params) < s:
        raise ValueError("Not enough parameters allocated to a_{i,i-1}")

    for i in range(s):
        for j in range(i):
            if j == i - 1:
                a_vars[(i, j)] = a_params[i]  # one per row
            else:
                a_vars[(i, j)] = b_vars[j]  # tied to corresponding b_j

    # ---------------------
    # STEP 3: c_i values
    # ---------------------
    c_vars = {i: sum(a_vars[(i, j)] for j in range(i)) for i in range(s)}

    # ---------------------
    # STEP 4: Order conditions (up to 3rd order)
    # ---------------------
    eqs = []
    eqs.append(sum(b_vars) - 1)
    eqs.append(sum(b_vars[i] * c_vars[i] for i in range(s)) - sp.Rational(1, 2))
    eqs.append(sum(b_vars[i] * c_vars[i]**2 for i in range(s)) - sp.Rational(1, 3))
    eqs.append(sum(b_vars[i] * sum(a_vars[(i, j)] * c_vars[j] for j in range(i)) for i in range(s)) - sp.Rational(1, 6))

    # ---------------------
    # STEP 5: Jacobian
    # ---------------------
    M = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in eqs])
    rank = M.rank()
    dof = len(params) - rank

    return {
        "Stages (s)": s,
        "2s - 7 Parameters": num_params,
        "Rooted Tree Conditions": len(eqs),
        "Jacobian Rank": rank,
        "Degrees of Freedom": dof,
        "Matches 2s - 7?": rank == 4
    }

# Test for s = 15
#can change for too verify for any S compute time increases though alot
print(verify_2s_minus_7_matches_rooted_trees(15))
