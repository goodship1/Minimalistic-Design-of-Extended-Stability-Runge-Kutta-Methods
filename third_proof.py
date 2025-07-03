import sympy as sp

def verify_two_register_order3(S, verbose=False):
    """
    Verifies that a 2‑register Van‑der‑Houwen tableau with (2S‑7) parameters
    can satisfy all third‑order rooted‑tree conditions and that the Jacobian
    wrt those parameters has full rank 4.

    Returns (rank, dof).
    """
    assert S >= 8, "Need at least S = 8 stages"

    n_params = 2 * S - 7
    params   = sp.symbols(f'p0:{n_params}')

    # -------- b_i pattern (shared for first 8 stages) ------------------------
    b_shared = params[0]
    b_unique = params[1 : 1 + (S - 8)]
    b = [b_shared]*8 + list(b_unique)        # length S
    b = b[:S]                                # in case S == 8

    # -------- a_{i,i-1} diagonals -------------------------------------------
    a_diag   = params[1 + (S - 8):]          # length S
    assert len(a_diag) == S

    # Construct full lower‑triangular a (only strict lower needed)
    a = {}
    for i in range(S):
        for j in range(i):
            if j == i - 1:
                a[(i, j)] = a_diag[i]
            else:
                a[(i, j)] = b[j]             # tie to same b_j

    # -------- c_i (row sums of a, strict lower) ------------------------------
    c = [sum(a[(i, j)] for j in range(i)) for i in range(S)]

    # -------- 3rd‑order rooted‑tree conditions ------------------------------
    conds = []
    conds.append(sum(b) - 1)
    conds.append(sum(b[i] * c[i]           for i in range(S)) - sp.Rational(1,2))
    conds.append(sum(b[i] * c[i]**2        for i in range(S)) - sp.Rational(1,3))
    conds.append(sum(b[i] * sum(a[(i,j)]*c[j] for j in range(i)) for i in range(S))
                 - sp.Rational(1,6))

    # Jacobian and rank
    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])
    rank = J.rank()
    dof  = n_params - rank

    if verbose:
        print(f"S = {S}")
        print("Number params :", n_params)
        print("Jacobian rank :", rank)
        print("DoF remaining :", dof)
        print()

    return rank, dof

# demo for two stage counts
for S in [15, 21]:
    verify_two_register_order3(S, verbose=True)
