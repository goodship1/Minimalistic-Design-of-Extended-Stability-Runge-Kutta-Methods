import sympy as sp

def verify_two_register_order3(S, verbose=False):
    """
    Verifies that a 2-register Van-der-Houwen tableau with (2S - 8) parameters
    can satisfy all third-order rooted-tree conditions and that the Jacobian
    wrt those parameters has full rank 4.

    Returns (rank, dof, Jacobian matrix, params).
    """
    assert S >= 8, "Need at least S = 8 stages"

    n_params = 2 * S - 8
    params = sp.symbols(f'p0:{n_params}')

    # -------- b_i pattern (shared for first 8 stages) ------------------------
    b_shared = params[0]
    b_unique = params[1 : 1 + (S - 8)]
    b = [b_shared]*8 + list(b_unique)
    b = b[:S]

    # -------- a_{i,i-1} diagonals -------------------------------------------
    a_diag = params[1 + (S - 8):]
    assert len(a_diag) == S - 1

    # Construct full lower-triangular a (strict lower part)
    a = {}
    for i in range(S):
        for j in range(i):
            if j == i - 1 and i > 0:
                a[(i, j)] = a_diag[i - 1]  # Correct indexing for a_diag
            else:
                a[(i, j)] = b[j]

    # -------- c_i (row sums of a) -------------------------------------------
    c = [sum(a[(i, j)] for j in range(i)) for i in range(S)]

    # -------- 3rd-order rooted-tree order conditions ------------------------
    conds = []
    conds.append(sum(b) - 1)
    conds.append(sum(b[i] * c[i] for i in range(S)) - sp.Rational(1,2))
    conds.append(sum(b[i] * c[i]**2 for i in range(S)) - sp.Rational(1,3))
    conds.append(sum(b[i] * sum(a[(i,j)] * c[j] for j in range(i)) for i in range(S)) - sp.Rational(1,6))

    # Build Jacobian matrix of derivatives wrt params
    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])

    rank = J.rank()
    dof = n_params - rank

    if verbose:
        print(f"S = {S}")
        print("Number of parameters:", n_params)
        print("Jacobian rank:", rank)
        print("Degrees of freedom remaining:", dof)
        print("\nJacobian matrix J (symbolic):")
        sp.pprint(J)
        print("\nColumns corresponding to b_shared and b_unique parameters:")

        for col_idx in range(0, 1 + (S - 8)):
            print(f"\nColumn {col_idx} (parameter {params[col_idx]}):")
            sp.pprint(J[:, col_idx])

    return rank, dof, J, params

if __name__ == "__main__":
    # Example: check for S=15 and S=21 with verbose output
    for S in [16,17,18,19,20,22]:
        verify_two_register_order3(S, verbose=True)

    # Uncomment to check a range of S values (less verbose)
    # for S in range(15, 51):
    #     rank, dof, _, _ = verify_two_register_order3(S)
    #     print(f"S={S}, rank={rank}, dof={dof}")
