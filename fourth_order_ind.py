#!/usr/bin/env python3
# zero_weight_sweep_order4.py
# Verify 4th-order preservation under zero-weight extension from an S=16 base,
# and explicitly assert the 2S-8 Van-der-Houwen structure at each S.

import numpy as np

# ---- Your S=16 base (A,b) ----
a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.297950632696351, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.522026933033341, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.144349746352280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.000371956295732390, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.124117473662160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.192800131150961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.00721201688860849, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.385496874023061, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.248192855959921, 0, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, -4.25371891111175e-5, 0, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, 0.138371044215410, 0, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.403108090476214, 0, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, 0.125164780662438, 0, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, -0.0422844329611440, -0.00579862710501764, 0, 0],
    [0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344, -0.295854950063981, 0.163017169512979, -0.0819824325549522, 0.546008221888163, -0.0422844329611440, 0.467431197768081, 0.502036131647685, 0]
], dtype=float)

b = np.array([
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    0.0892840764249344, 0.0892840764249344, 0.0892840764249344, 0.0892840764249344,
    -0.29585495006398077, 0.1630171695129791, -0.08198243255495223, 0.5460082218881631,
    -0.04228443296114401, 0.46743119776808084, -0.45495020324595, -0.01565718174267131
], dtype=float)
# -----------------------------------------

# ---------- 2S-8 / VdH checks ----------

def is_strictly_lower_triangular(A, tol=1e-12) -> bool:
    return np.all(np.triu(np.abs(A), k=0) < tol)

def vdh_ties_ok(A: np.ndarray, b: np.ndarray, tol=1e-12) -> bool:
    """A[i,j] == b[j] for all j < i-1 (VdH ties)."""
    S = len(b)
    for i in range(S):
        for j in range(i-1):
            if abs(A[i, j] - b[j]) > tol:
                return False
    return True

def assert_2S_minus_8(A: np.ndarray, b: np.ndarray, S_expect=None, name=""):
    S = len(b)
    if S_expect is not None:
        assert S == S_expect, f"{name}: S mismatch (got {S}, expected {S_expect})"
    # (i) explicit RK
    assert is_strictly_lower_triangular(A), f"{name}: A is not strictly lower-triangular"
    # (ii) first 8 b's tied
    assert np.allclose(b[:8], b[0], atol=1e-12), f"{name}: b[0..7] must be equal (2S-8 tie)"
    # (iii) VdH ties
    assert vdh_ties_ok(A, b), f"{name}: VdH ties broken (A[i,j] != b[j] for some j<i-1)"
    # (iv) parameter count info (for logging)
    n_params = 2*S - 8
    print(f"{name}: conforms to 2S-8 (S={S}, params={n_params})")

# ---------- Order-4 residuals ----------

def order4_residuals(A: np.ndarray, b: np.ndarray):
    """Return dict of residuals for the 8 fourth-order conditions and the max abs residual."""
    c  = A.sum(axis=1)
    Ac = A @ c
    Ac2 = A @ (c**2)
    A2c = A @ Ac
    targets = dict(b1=1.0, bc=0.5, bc2=1/3, bAc=1/6, bc3=1/4, bAc_c=1/8, bA_c2=1/12, bA2c=1/24)
    vals = dict(
        b1    = float(b @ np.ones_like(b)),
        bc    = float(b @ c),
        bc2   = float(b @ (c**2)),
        bAc   = float(b @ Ac),
        bc3   = float(b @ (c**3)),
        bAc_c = float(b @ (Ac * c)),
        bA_c2 = float(b @ Ac2),
        bA2c  = float(b @ A2c),
    )
    res = {k: vals[k] - targets[k] for k in targets}
    return vals, res, max(abs(v) for v in res.values())

# ---------- Zero-weight extension (VdH + 2S-8) ----------

def extend_zero_weight(A: np.ndarray, b: np.ndarray, d_new: float | None = None):
    """
    Append stage S+1 with b_{S+1}=0 (free new weight but chosen as zero).
    Freeze all old entries. Fill ONLY the new last row:
      A[S, :S-1] = b[:S-1]   (VdH ties)
      A[S, S-1]  = d_new     (new subdiagonal; arbitrary choice)
    """
    S = len(b)
    A2 = np.zeros((S+1, S+1), dtype=float)
    A2[:S, :S] = A  # freeze old block exactly
    b2 = np.concatenate([b, [0.0]])
    if S >= 1:
        A2[S, :S-1] = b[:S-1]
        if d_new is None:
            d_new = b[S-1]  # any reasonable default; order unaffected since b_{S}=0
        A2[S, S-1] = d_new
    return A2, b2

# ---- Base: assert 2S-8 and order-4 ----
assert_2S_minus_8(a, b, S_expect=16, name="Base S=16")
vals, res, worst = order4_residuals(a, b)
print("Base S=16 order-4 residuals: max|residual| =", f"{worst:.3e}")

# ---- Zero-weight extension sweep with explicit 2S-8 assertions ----
S0 = len(b)
Smax = 300 # adjust as needed
Acur, bcur = a.copy(), b.copy()
print("\nZero-weight extension sweep (freeze old, append b_new=0):")
print("   S    max|residual|")
for S in range(S0, Smax+1):
    while len(bcur) < S:
        Acur, bcur = extend_zero_weight(Acur, bcur, d_new=bcur[-1])
    # 2S-8 check at this S
    assert_2S_minus_8(Acur, bcur, S_expect=S, name=f"S={S}")
    # order-4 residuals
    _, _, worstS = order4_residuals(Acur, bcur)
    print(f"  {S:2d}    {worstS:.3e}")

# Optional hard assertion: everything should be machine-precision
# assert all(order4_residuals(*extend_zero_weight(a.copy(), b.copy(), d_new=b[-1]))[2] < 1e-12 for _ in range(1))
