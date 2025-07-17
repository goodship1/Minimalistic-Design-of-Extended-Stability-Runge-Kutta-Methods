# Re-import libraries and define the function again after code reset

import sympy as sp
import random

def order4_symbolic_rank_generic(S, seed=0, verbose=False):
    n_params = 2*S - 8
    params = sp.symbols(f'p0:{n_params}')

    b_shared = params[0]
    b_unique = params[1:1+(S-8)]
    b_syms = [b_shared if i < 8 else b_unique[i-8] for i in range(S)]

    a_diag = params[1+(S-8):]
    a_syms = {}
    for i in range(S):
        for j in range(i):
            if j == i-1:
                a_syms[(i,j)] = a_diag[i-1]
            else:
                a_syms[(i,j)] = b_syms[j]

    c_syms = [sum(a_syms[(i,j)] for j in range(i)) for i in range(S)]

    conds = [
        sum(b_syms) - 1,
        sum(b_syms[i]*c_syms[i] for i in range(S)) - sp.Rational(1,2),
        sum(b_syms[i]*c_syms[i]**2 for i in range(S)) - sp.Rational(1,3),
        sum(b_syms[i]*sum(a_syms[(i,j)]*c_syms[j] for j in range(i)) for i in range(S)) - sp.Rational(1,6),
        sum(b_syms[i]*c_syms[i]**3 for i in range(S)) - sp.Rational(1,4),
        sum(b_syms[i]*sum(a_syms[(i,j)]*c_syms[j]**2 for j in range(i)) for i in range(S)) - sp.Rational(1,8),
        sum(b_syms[i]*c_syms[i]*sum(a_syms[(i,j)]*c_syms[j] for j in range(i)) for i in range(S)) - sp.Rational(1,12),
        sum(b_syms[i]*sum(a_syms[(i,j)]*sum(a_syms[(j,k)]*c_syms[k] for k in range(j)) for j in range(i)) for i in range(S)) - sp.Rational(1,24)
    ]

    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])

    rng = random.Random(seed)
    subs = {p: sp.Rational(rng.randint(1, 5), rng.randint(1, 5)) for p in params}
    J_num = J.subs(subs)
    rank = J_num.rank()

    if verbose:
        print(f"S={S}, rank={rank}")

    return rank

# Run from S=17 to S=50
results = []
for S in range(17, 51):
    r = order4_symbolic_rank_generic(S, seed=42)
    results.append((S, r))

print(results)
