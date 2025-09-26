#!/usr/bin/env python3
# zero_weight_sweep_order3_third15.py
# Verify: (i) VdH (2-register LSRK) ties, (ii) 2S-8 compressed pattern,
# (iii) order-3 conditions, (iv) zero-weight extension preserves order and R(z).

import numpy as np

# ---------- Your 15-stage, 3rd-order ESRK tableau (A,b) ----------
A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0],
    [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0]
], dtype=float)

b = np.array([
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.035898932499408134,
    0.035898932499408134, 0.035898932499408134, 0.006612457947210495,
    0.21674686949693006, 0.0, 0.42264597549826616,
    0.03276149074985981, 0.0330623263939421, 0.0009799086295048407
], dtype=float)

# ---------- Utilities: structure checks ----------

def is_strictly_lower_triangular(A, tol=1e-12):
    return np.all(np.triu(np.abs(A), k=0) < tol)

def vdh_ties_ok(A: np.ndarray, b: np.ndarray, tol=1e-12) -> bool:
    """Van-der-Houwen ties: A[i,j] == b[j] for all j < i-1."""
    S = len(b)
    for i in range(S):
        for j in range(i-1):
            if abs(A[i, j] - b[j]) > tol:
                return False
    return True

def assert_2S_minus_8(A: np.ndarray, b: np.ndarray, name=""):
    S = len(b)
    assert is_strictly_lower_triangular(A), f"{name}: A not strictly lower triangular"
    # In your compressed family, the first 8 weights are equal
    assert np.allclose(b[:8], b[0], atol=1e-12), f"{name}: b[0..7] must be equal (2S-8 tie)"
    # VdH ties ensure true 2-register LSRK
    assert vdh_ties_ok(A, b), f"{name}: VdH ties broken (A[i,j] != b[j] for some j<i-1)"
    print(f"{name}: conforms to 2S-8 (S={S}, params={2*S-8})")

# ---------- Order-3 residuals ----------
# Conditions (explicit RK):
#   1) sum b_i = 1
#   2) b^T c = 1/2
#   3) b^T c^2 = 1/3
#   4) b^T A c = 1/6
def order3_residuals(A: np.ndarray, b: np.ndarray):
    c  = A.sum(axis=1)
    Ac = A @ c
    targets = dict(b1=1.0, bc=0.5, bc2=1/3, bAc=1/6)
    vals = dict(
        b1   = float(b @ np.ones_like(b)),
        bc   = float(b @ c),
        bc2  = float(b @ (c**2)),
        bAc  = float(b @ Ac),
    )
    res = {k: vals[k] - targets[k] for k in targets}
    worst = max(abs(v) for v in res.values())
    return vals, res, worst

# ---------- Zero-weight extension (append stage with b_new=0) ----------

def extend_zero_weight(A: np.ndarray, b: np.ndarray, d_new: float | None = None):
    """
    Append stage S+1 with b_{S+1}=0 and Van-der-Houwen ties:
      A[S, :S-1] = b[:S-1],  A[S, S-1] = d_new  (subdiagonal)
    All previous entries frozen exactly.
    """
    S = len(b)
    A2 = np.zeros((S+1, S+1), dtype=float)
    A2[:S, :S] = A
    b2 = np.concatenate([b, [0.0]])
    if S >= 1:
        A2[S, :S-1] = b[:S-1]
        if d_new is None:
            d_new = b[S-1]           # any default; order unaffected since weight=0
        A2[S, S-1] = d_new
    return A2, b2

# ---------- Numerical stability function R(z) evaluator ----------
def R_eval(A: np.ndarray, b: np.ndarray, z: complex) -> complex:
    """
    Evaluate stability function R(z) by applying one RK step to y' = (z/h) y
    in the normalized form using u_i = k_i / λ:
        u_i = 1 + z * sum_{j<i} a_{ij} u_j
        R(z) = 1 + z * sum_i b_i u_i
    This depends only on (A,b) and z.
    """
    S = len(b)
    u = np.zeros(S, dtype=complex)
    for i in range(S):
        u[i] = 1.0 + z * np.dot(A[i, :i], u[:i])
    return 1.0 + z * np.dot(b.astype(complex), u)

def stability_match(A_old, b_old, A_new, b_new, Z_samples, tol=1e-12):
    """
    Check max |R_new(z)-R_old(z)| over sample points Z_samples.
    """
    diffs = []
    for z in Z_samples:
        r_old = R_eval(A_old, b_old, z)
        r_new = R_eval(A_new, b_new, z)
        diffs.append(abs(r_new - r_old))
    return max(diffs), diffs

# ----------------------- Main: run the sweep -----------------------

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("Base S=15 (p=3) checks:")
    assert_2S_minus_8(A, b, name="Base S=15")
    _, _, worst = order3_residuals(A, b)
    print("Base S=15 order-3 residuals: max|residual| =", f"{worst:.3e}")

    # Sweep: repeatedly append b_new=0 stages, keep VdH ties, verify order & R(z)
    S0 = len(b)
    Smax = 300  # adjust as you like
    Acur, bcur = A.copy(), b.copy()

    # A few test points for R(z) along negative real and imaginary axes
    Z_samples = [0, -0.1, -0.5, -1.0, -2.0, 0.25j, 0.5j, 1j, -1+0.5j, -2+1j]

    print("\nZero-weight extension sweep (VdH ties, freeze old):")
    print("   S    max|residual|   max|ΔR(z)|")
    for S in range(S0, Smax + 1):
        # Extend up to exactly S
        while len(bcur) < S:
            Acur, bcur = extend_zero_weight(Acur, bcur, d_new=bcur[-1])

        # Structure checks
        assert_2S_minus_8(Acur, bcur, name=f"S={S}")

        # Order-3 residuals
        _, _, worstS = order3_residuals(Acur, bcur)

        # Stability function equality vs original base
        maxdiff, _ = stability_match(A, b, Acur, bcur, Z_samples, tol=1e-12)

        print(f"  {S:3d}   {worstS:.3e}     {maxdiff:.3e}")

    print("\nAll done.")
