import sympy as sp
import numpy as np
from scipy.optimize import approx_fprime

def build_full_sixth_order_eqs(s, params, k):
    """
    Build all 30 sixth-order rooted-tree equations for the 2S-k Van der Houwen pattern:
      • b[0..k-1] share p0
      • b[k..s-1] are independent p1..p_{s-k}
      • a[i,i-1] are p_{...}, all other a[i,j< i-1] = b_j
    """
    # 1) b vector
    b_shared = params[0]
    b_unique = params[1:1 + (s - k)]
    b = [b_shared if i < k else b_unique[i - k] for i in range(s)]

    # 2) A-matrix entries
    a_params = params[1 + (s - k):]
    a = {}
    # we have exactly s-1 free subdiagonals,
    # so len(a_params) should be s-1
    for i in range(1, s):
        for j in range(i):
            if j == i-1:
                a[(i,j)] = a_params[i-1]
            else:
                a[(i,j)] = b[j]

    # 3) c times
    c = {i: sum(a[(i,j)] for j in range(i)) for i in range(s)}

    # 4) build the 30 rooted-tree conditions T1..T30
    eqs = []
    # T1–T6: moments c^m
    eqs += [sum(b[i]*c[i]**m for i in range(s)) - sp.Rational(1, m+1)
             for m in range(0,6)]

    # T7..T10: one A-layer
    eqs.append(sum(b[i]*sum(a[(i,j)]*c[j]**4 for j in range(i)) for i in range(s)) - sp.Rational(1,15))
    eqs.append(sum(b[i]*c[i]*sum(a[(i,j)]*c[j]**3 for j in range(i)) for i in range(s)) - sp.Rational(1,30))
    eqs.append(sum(b[i]*c[i]**2*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1,36))
    eqs.append(sum(b[i]*c[i]**3*sum(a[(i,j)]*c[j]   for j in range(i)) for i in range(s)) - sp.Rational(1,45))

    # T11..T14: two A-layers
    eqs.append(sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*c[k]**3 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,60))
    eqs.append(sum(b[i]*c[i]*sum(a[(i,j)]*sum(a[(j,k)]*c[k]**2 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,60))
    eqs.append(sum(b[i]*c[i]**2*sum(a[(i,j)]*sum(a[(j,k)]*c[k]   for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,60))
    eqs.append(sum(b[i]*sum(a[(i,j)]*c[j]**2*sum(a[(j,k)]*c[k]   for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,60))

    # T15, T16: three A-layers
    eqs.append(sum(
        b[i]
        * sum(
            a[(i,j)]
            * sum(
                a[(j,k)]
                * sum(a[(k,l)]*c[l]**2 for l in range(k))
                for k in range(j)
              )
            for j in range(i)
          )
        for i in range(s)
    ) - sp.Rational(1,120))

    eqs.append(sum(
        b[i]
        * sum(
            a[(i,j)]
            * sum(
                a[(j,k)]
                * c[k]
                * sum(a[(k,l)]*c[l] for l in range(k))
                for k in range(j)
              )
            for j in range(i)
          )
        for i in range(s)
    ) - sp.Rational(1,120))

    # T17..T20
    eqs += [
      sum(b[i]*c[i]*sum(a[(i,j)]*c[j]*sum(a[(j,k)]*c[k] for k in range(j)) 
                     for j in range(i)) for i in range(s)) - sp.Rational(1,90),
      sum(b[i]*sum(a[(i,j)]*c[j]*sum(a[(j,k)]*c[k]**2 for k in range(j)) 
               for j in range(i)) for i in range(s)) - sp.Rational(1,90),
      sum(b[i]*c[i]**2*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1,60),
      sum(b[i]*c[i]*sum(a[(i,j)]*c[j]**3 for j in range(i)) for i in range(s)) - sp.Rational(1,45)
    ]

    # T21..T24
    eqs += [
      sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*sum(a[(k,l)]*c[l] 
               for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,120),

      sum(b[i]*sum(a[(i,j)]*c[j]**2*sum(a[(j,k)]*c[k] 
               for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,60),

      sum(b[i]*c[i]*sum(a[(i,j)]*sum(a[(j,k)]*c[k]**2 
               for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,60),

      sum(b[i]*sum(a[(i,j)]*c[j]*sum(a[(j,k)]*c[k]**2 
               for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,60),
    ]

    # T25..T28
    eqs += [
      sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*sum(a[(k,l)]*c[l] 
               for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,90),

      sum(b[i]*c[i]**3*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s)) 
      - sp.Rational(1,45),

      sum(b[i]*sum(a[(i,j)]*c[j]*sum(a[(j,k)]*c[k] for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,90),

      sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*sum(a[(k,l)]*c[l] 
               for l in range(k)) for k in range(j)) for j in range(i)) for i in range(s)) 
      - sp.Rational(1,720),
    ]

    # T29
    eqs.append(
      sum(b[i]*c[i]**4*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s))
      - sp.Rational(1,30)
    )

    # T30
    eqs.append(
      sum(
        b[i]
        * sum(
            a[(i,j)]
            * sum(
                a[(j,k)]
                * sum(
                    a[(k,l)]
                    * sum(a[(l,m)]*c[m] for m in range(l))
                    for l in range(k)
                  )
                for k in range(j)
              )
            for j in range(i)
          )
        for i in range(s)
      )
      - sp.Rational(1,120)
    )

    assert len(eqs) == 30
    return eqs

def verify_rank_2s_k(s, k, nprobe=5, eps=1e-6):
    """Build and test numeric rank of the 30×(2S−k) Jacobian for given (s,k)."""
    num_params = 2*s - k
    params     = [sp.Symbol(f'p{i}') for i in range(num_params)]
    eqs        = build_full_sixth_order_eqs(s, params, k)
    f          = sp.lambdify(params, eqs, "numpy")

    best = 0
    for _ in range(nprobe):
        x0 = np.random.rand(num_params)
        # build Jacobian rows by FD:
        J  = np.vstack([
            approx_fprime(x0, lambda x, i=i: f(*x)[i], eps)
            for i in range(30)
        ])
        best = max(best, np.linalg.matrix_rank(J))
    print(f"S={s}, k={k:2d}: params={num_params:2d}, rank={best:2d}/30")

if __name__ == "__main__":
    # example: test all k=1..6 for s=25
    for k in range(1,7):
        verify_rank_2s_k(s=25, k=k)
