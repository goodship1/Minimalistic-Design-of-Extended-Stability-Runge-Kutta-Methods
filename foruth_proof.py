def order4_rank_and_minimality(S, seed=0, verbose=False):
    n_params = 2*S - 7
    params   = sp.symbols(f'p0:{n_params}')

    # --- same pattern construction as before -------------------------------
    b_shared = params[0]
    b_unique = params[1:1+(S-8)]
    b = ([b_shared]*8 + list(b_unique))[:S]
    a_diag   = params[1+(S-8):]
    a = {(i,j): (a_diag[i] if j==i-1 else b[j]) for i in range(S) for j in range(i)}
    c = [sum(a[(i,j)] for j in range(i)) for i in range(S)]

    # 8 order conditions (trees up to order-4)
    conds = [
        sum(b) - 1,
        sum(b[i]*c[i]                        for i in range(S)) - sp.Rational(1,2),
        sum(b[i]*c[i]**2                    for i in range(S)) - sp.Rational(1,3),
        sum(b[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(S)) - sp.Rational(1,6),
        sum(b[i]*c[i]**3                   for i in range(S)) - sp.Rational(1,4),
        sum(b[i]*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(S)) - sp.Rational(1,8),
        sum(b[i]*c[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(S)) - sp.Rational(1,12),
        sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*c[k] for k in range(j)) for j in range(i)) for i in range(S)) - sp.Rational(1,24)
    ]

    # --- Jacobian -----------------------------------------------------------
    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])

    # evaluate at a random rational point to avoid enormous symbolic matrices
    rng   = random.Random(seed)
    subs  = {p: sp.Rational(rng.randint(1,5), rng.randint(1,5)) for p in params}
    rank  = J.subs(subs).rank()           # numeric but exact (rationals)

    if verbose:
        print(f"S = {S} | params = {n_params} | rank = {rank} | dof = {n_params-rank}")

    return rank
