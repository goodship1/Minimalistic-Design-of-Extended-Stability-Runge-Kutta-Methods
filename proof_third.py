import sympy as sp

# ---------------------------------------------------------------------------
# Symbolic VdH parameter builders
# ---------------------------------------------------------------------------
def vdh_params(S, prefix=None):
    """
    Build symbolic structures for the 2-register Van-der-Houwen tying pattern.

    Returns dict:
      params    : tuple of all parameter symbols (length 2S-8)
      b_shared  : shared weight symbol
      b_unique  : list length S-8 (may be 0 if S<=8)
      d_diag    : list length S-1 (subdiagonals a_{i,i-1})
      b         : list length S of stage weights
      a         : dict (i,j)->sym
      c         : list length S abscissae
      R         : 4-vector of order-3 residual expressions (==0 when satisfied)
    """
    if prefix is None:
        prefix = f"S{S}_"
    n_params = 2*S - 8
    params = sp.symbols(f'{prefix}p0:{n_params}', real=True)

    b_shared = params[0]
    b_unique = params[1:1+(S-8)]
    d_diag   = params[1+(S-8):]  # S-1 entries

    # stage weights
    b = [b_shared]*min(8,S) + list(b_unique)
    b = b[:S]

    # a dict: tied columns except subdiagonal
    a = {}
    for i in range(S):
        for j in range(i):
            if j == i-1:
                a[(i,j)] = d_diag[i-1]
            else:
                a[(i,j)] = b[j]

    # abscissae
    c = []
    for i in range(S):
        if i == 0:
            c.append(sp.Integer(0))
        else:
            c.append(sp.Add(*[a[(i,j)] for j in range(i)]))

    # order-3 residuals
    OC1 = sp.Add(*b) - sp.Integer(1)
    OC2 = sp.Add(*[b[i]*c[i] for i in range(S)]) - sp.Rational(1,2)
    OC3 = sp.Add(*[b[i]*c[i]**2 for i in range(S)]) - sp.Rational(1,3)
    OC4 = sp.Integer(0)
    for i in range(S):
        OC4 += b[i]*sp.Add(*[a[(i,j)]*c[j] for j in range(i)])
    OC4 -= sp.Rational(1,6)

    R = sp.Matrix([sp.simplify(OC1), sp.simplify(OC2), sp.simplify(OC3), sp.simplify(OC4)])

    return dict(
        params=params,
        b_shared=b_shared,
        b_unique=list(b_unique),
        d_diag=list(d_diag),
        b=b,
        a=a,
        c=c,
        R=R,
    )

# ---------------------------------------------------------------------------
# Build S=15 and S=16 symbolic systems
# ---------------------------------------------------------------------------
S0 = 15
S1 = 16
vdh15 = vdh_params(S0, prefix="S15_")
vdh16 = vdh_params(S1, prefix="S16_")

# ---------------------------------------------------------------------------
# Build substitution map: map the first 15-stage structure of S=16
# onto the S=15 symbols, leaving the NEW stage (index 15) free.
# ---------------------------------------------------------------------------
subs_map = {}

# shared weight
subs_map[vdh16['b_shared']] = vdh15['b_shared']

# copy the first (S0-8)=7 unique weights
for k in range(S0-8):
    subs_map[vdh16['b_unique'][k]] = vdh15['b_unique'][k]

# identify the NEW weight (stage index 15, in S=16 system)
b_new = vdh16['b_unique'][S0-8]  # b_unique[7]

# copy subdiagonals for old stages (S0-1)=14 of them
for k in range(S0-1):
    subs_map[vdh16['d_diag'][k]] = vdh15['d_diag'][k]

# identify NEW subdiagonal (for last stage)
d_new = vdh16['d_diag'][S0-1]

# ---------------------------------------------------------------------------
# Substitute and compute difference R16 - R15
# ---------------------------------------------------------------------------
R16_mapped = sp.simplify(vdh16['R'].subs(subs_map))
R15 = vdh15['R']
Rdiff = sp.simplify(R16_mapped - R15)

print("==== Symbolic difference R16 - R15 (before factoring) ====")
for k in range(4):
    print(f"Comp {k+1}:")
    print("  ", sp.simplify(Rdiff[k]))
print()

# ---------------------------------------------------------------------------
# Show that each component factors through b_new
# ---------------------------------------------------------------------------
print("==== Factor each component of (R16 - R15) by the new stage weight b_new ====")
print("b_new symbol:", b_new)
for k in range(4):
    factored = sp.factor(Rdiff[k], b_new, deep=True)
    print(f"Comp {k+1} factored:")
    print("  ", factored)
    # also show quotient if divisible
    quotient = sp.simplify(sp.cancel(Rdiff[k] / b_new))
    print("    quotient =", quotient)
print()

# ---------------------------------------------------------------------------
# Verify that setting b_new = 0 makes Rdiff = 0 (identity)
# ---------------------------------------------------------------------------
zero_test = sp.simplify(Rdiff.subs({b_new: sp.Integer(0)}))
print("==== Rdiff with b_new=0 ====")
for k in range(4):
    print(f"Comp {k+1} ->", zero_test[k])
print()

# ---------------------------------------------------------------------------
# Optional: verify the *new* stage's contribution has the expected forms
#           [b_new,
#            b_new * c_new,
#            b_new * c_new**2,
#            b_new * sum_j a_{new,j} * c_j_old]
# We'll build those expressions explicitly and compare.
# ---------------------------------------------------------------------------
# old c's (S=15)
c_old = vdh15['c']
# new stage in S=16 has index S0 (15)
c_new = vdh16['c'][S0]
expr_expected = sp.Matrix([
    b_new,
    b_new * c_new,
    b_new * c_new**2,
    b_new * sp.Add(*[vdh16['a'][(S0,j)] * c_old[j] for j in range(S0)])  # only old c's (careful!)
])
expr_expected_mapped = sp.simplify(expr_expected.subs(subs_map))  # map old param syms

print("==== Expected new-stage contribution (mapped to old symbols) ====")
for k in range(4):
    print(f"Comp {k+1} expected:", expr_expected_mapped[k])

print("\n==== Compare (R16 - R15) vs expected new-stage contribution ====")
for k in range(4):
    diff_cmp = sp.simplify(Rdiff[k] - expr_expected_mapped[k])
    print(f"Comp {k+1} difference:", diff_cmp)
