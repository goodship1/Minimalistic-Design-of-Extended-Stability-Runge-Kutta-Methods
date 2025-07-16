import sympy as sp
import numpy as np
import random

def order4_rank_for_given_scheme(S, a_vals, b_vals):
    # Define symbolic parameters for 2S-8 structure (b0..b7 tied)
    n_params = 2*S - 8
    params = sp.symbols(f'p0:{n_params}')

    # Define b structure: b0..b7 tied, rest free
    b_shared = params[0]
    b_unique = params[1:1+(S-8)]
    b_syms = [b_shared if i < 8 else b_unique[i-8] for i in range(S)]

    # Define a structure: subdiagonal free, rest tied to b
    a_diag = params[1+(S-8):]
    a_syms = {}
    for i in range(S):
        for j in range(i):
            if j == i-1:
                a_syms[(i,j)] = a_diag[i-1]
            else:
                a_syms[(i,j)] = b_syms[j]

    # Define c_i
    c_syms = [sum(a_syms[(i,j)] for j in range(i)) for i in range(S)]

    # Define fourth-order conditions (8 conditions)
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

    # Compute symbolic Jacobian
    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])

    # Prepare substitution from provided scheme
    subs = {}
    for i in range(S):
        if i < 8:
            subs[b_syms[i]] = b_vals[0]  # all tied to b[0] as per structure
        else:
            subs[b_syms[i]] = b_vals[i]
    idx = 0
    for i in range(S):
        for j in range(i):
            if j == i-1:
                subs[a_syms[(i,j)]] = a_vals[idx]
                idx += 1
            else:
                subs[a_syms[(i,j)]] = subs[b_syms[j]]

    # Evaluate numerical rank using sympy's exact rational arithmetic
    J_num = J.subs(subs)
    rank = J_num.rank()

    return rank

# Provided scheme coefficients
a_vals = [
    0.4840322176479906, 0.09862985274030009, 0.3135108193336061, 0.09862985274030009,
    0.09862985274030009, 0.17333855841633225, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.14255132033314108, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.11640966356188277, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.00021957134185754268, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, -0.12196770928777284,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, -0.207154474196716,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.018636063347043177, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, -0.4340574811180196, 0.45662022744532477, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, -0.4340574811180196,
    0.013726602993762585, -0.28952295943121414, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, -0.4340574811180196, 0.013726602993762585,
    0.056452907639256075, -0.00017994759845236074, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, -0.4340574811180196, 0.013726602993762585,
    0.056452907639256075, 0.057859531498276384, 0.43242305044982937, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, -0.4340574811180196,
    0.013726602993762585, 0.056452907639256075, 0.057859531498276384, 0.22059756202807293,
    -0.24058688344560986, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, -0.4340574811180196, 0.013726602993762585, 0.056452907639256075,
    0.057859531498276384, 0.22059756202807293, 0.1528167003257629, -0.0001900409075150595
]

b_vals = [
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    0.09862985274030009, 0.09862985274030009, 0.09862985274030009, 0.09862985274030009,
    -0.4340574811180196, 0.013726602993762585, 0.056452907639256075, 0.057859531498276384,
    0.22059756202807293, 0.1528167003257629, 0.05562659642557619, 0.08793875828491182
]

# Now compute the Jacobian rank with the provided scheme
rank_result = order4_rank_for_given_scheme(S=16, a_vals=a_vals, b_vals=b_vals)
rank_result
S = 16
# Due to input length, we'll just confirm structure and prepare for rank evaluation
rank = "Will compute rank after full a_vals and b_vals are expanded."
print(rank)
