import sympy as sp
import numpy as np

def verify_2s_minus_6_matches_rooted_trees(s, nprobe=5):
    """
    Build the 2s-6 Van der Houwen template (fifth order) and check
    whether the 17 order-5 equations become independent (i.e. numeric Jacobian rank = 17).
    
    Template (zero-based indexing: stages 0..s-1):
      • b_shared = p0, used for b[0..6]
      • b_unique = p1..p[s-7], used for b[7..s-1]
        => total b-parameters = 1 + (s - 7) = s - 6
      • a_params = p[(s - 6) .. (s - 6) + s - 1]  (length s)
         – for i = 1..(s-1), a_{i, i-1} = a_params[i-1]  (these are s-1 subdiagonals)
         – one extra free coefficient a_{s-1, s-3} = a_params[s-1]
      • all other a_{i,j} (with j < i-1, j != s-3 when i=s-1) are forced equal to b[j].
    
    Returns a dict:
      {
        "Stages (s)": s,
        "2s - 6 Parameters": 2*s - 6,
        "Order Conditions": 17,
        "Count Test OK?": (2*s - 6 >= 17),
        "Numeric Rank Test?": True/False,
        "Numeric Rank": best observed numeric rank,
        "Structure": "2S−6 ESRK (Van der Houwen style)"
      }
    """
    assert s >= 9, "Need s ≥ 9 so that 2s-6 ≥ 17."
    num_params = 2*s - 6
    # 1) Build symbolic parameter list p0..p_{2s-7}  (total = 2s-6 symbols)
    params = [sp.Symbol(f'p{i}') for i in range(num_params)]
    
    # 2) b_i construction:
    #    p0 => b_shared (applies for i=0..6),
    #    p1..p[s-7] => b_unique for i=7..(s-1).
    b_shared = params[0]
    b_unique = params[1 : 1 + (s - 7)]   # length = s-7
    b_vars = []
    for i in range(s):
        if i < 7:
            b_vars.append(b_shared)
        else:
            b_vars.append(b_unique[i - 7])
    
    # 3) a_{i,j} construction:
    #    a_params = next s symbols: length = s
    a_params = params[ (1 + (s-7)) : ]  # this slice has length = num_params - (s-6) = (2s-6) - (s-6) = s
    #    Assign subdiagonal: a_{i, i-1} = a_params[i-1]  for i = 1..s-1
    #    Assign "extra free"   a_{(s-1),(s-3)} = a_params[s-1]
    #    Otherwise (j < i-1 or j = i-2 when i ≠ s-1), tie: a_{i,j} = b_vars[j]
    a_vars = {}
    for i in range(s):
        for j in range(i):
            if j == i - 1:
                # subdiagonal (i,i-1)
                a_vars[(i,j)] = a_params[i - 1]
            elif (i == s - 1) and (j == s - 3):
                # extra free entry at (s-1, s-3)
                a_vars[(i,j)] = a_params[s - 1]
            else:
                # forced tie to b_j
                a_vars[(i,j)] = b_vars[j]
    
    # 4) Build stage times c_i = sum_{j=0..i-1} a_{i,j}
    c_vars = { i: sum(a_vars[(i,j)] for j in range(i)) for i in range(s) }
    
    # 5) Form the 17 fifth-order rooted-tree equations (symbolic)
    b, c, a = b_vars, c_vars, a_vars
    eqs = []
    # order-1
    eqs.append( sum(b[i] for i in range(s)) - 1 )
    # order-2
    eqs.append( sum(b[i]*c[i] for i in range(s)) - sp.Rational(1,2) )
    # order-3
    eqs.append( sum(b[i]*c[i]**2 for i in range(s)) - sp.Rational(1,3) )
    eqs.append( sum( b[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s) ) - sp.Rational(1,6) )
    # order-4
    eqs.append( sum(b[i]*c[i]**3 for i in range(s)) - sp.Rational(1,4) )
    eqs.append( sum(b[i]*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1,8) )
    eqs.append( sum(b[i]*c[i]*sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s)) - sp.Rational(1,12) )
    eqs.append( sum(
                   b[i]
                   * sum( a[(i,j)] * sum( a[(j,k)]*c[k] for k in range(j) ) for j in range(i) )
                   for i in range(s)
                 ) - sp.Rational(1,24) )
    # order-5 (there are 17 of them in total)
    eqs.append( sum(b[i]*c[i]**4 for i in range(s)) - sp.Rational(1,5) )
    eqs.append( sum(b[i]*sum(a[(i,j)]*c[j]**3 for j in range(i)) for i in range(s)) - sp.Rational(1,10) )
    eqs.append( sum(b[i]*c[i]*sum(a[(i,j)]*c[j]**2 for j in range(i)) for i in range(s)) - sp.Rational(1,15) )
    eqs.append( sum(b[i]*sum(a[(i,j)]*sum(a[(j,k)]*c[k]**2 for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,20) )
    eqs.append( sum(b[i]*sum(a[(i,j)]*c[j]*sum(a[(j,k)]*c[k] for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,30) )
    eqs.append( sum(b[i]*c[i]**2 * sum(a[(i,j)]*c[j] for j in range(i)) for i in range(s)) - sp.Rational(1,20) )
    eqs.append( sum(b[i]*c[i] * sum(a[(i,j)]*sum(a[(j,k)]*c[k] for k in range(j)) for j in range(i)) for i in range(s)) - sp.Rational(1,40) )
    eqs.append( sum(
                   b[i]
                   * sum( a[(i,j)]
                          * sum( a[(j,k)]
                                 * sum( a[(k,l)]*c[l] for l in range(k) )
                                for k in range(j) )
                         for j in range(i) )
                   for i in range(s)
                 ) - sp.Rational(1,120) )
    # Now the 17th condition (sum_{i,j,k,l,m} b_i * a_{i,j}*a_{j,k}*a_{k,l}*a_{l,m} c[m] = 1/120).
    # We write it out explicitly in nested loops:
    term17 = sp.Integer(0)
    for i in range(s):
        for j in range(i):
            for k in range(j):
                for l in range(k):
                    for m in range(l):
                        term17 += b[i]*a[(i,j)]*a[(j,k)]*a[(k,l)]*a[(l,m)]*c[m]
    eqs.append( term17 - sp.Rational(1,120) )
    assert len(eqs) == 17

    # 6) Build the symbolic Jacobian matrix: 17 equations × (2s-6) parameters
    J_sym = sp.Matrix([[ sp.diff(eqs[i], params[j]) for j in range(num_params)] 
                       for i in range(17)])
    
    # 7) Numeric‐rank check: try `nprobe` random floats in (0,1).
    best_rank = 0
    for trial in range(nprobe):
        # pick a random assignment for each pᵢ in (0,1)
        subs = { params[j]: float(np.random.rand()) for j in range(num_params) }
        # evaluate the Jacobian numerically
        J_num = np.zeros((17, num_params), dtype=float)
        for i in range(17):
            for j in range(num_params):
                J_num[i,j] = float(J_sym[i,j].subs(subs))
        # compute numeric rank via SVD
        U, Svals, Vt = np.linalg.svd(J_num, full_matrices=False)
        tol = max(J_num.shape)*np.finfo(float).eps * Svals.max()
        rank = int((Svals > tol).sum())
        best_rank = max(best_rank, rank)
        print(f"  Trial {trial:2d}: numeric rank = {rank} / 17")
        if rank == 17:
            break  # no need for further probes

    passed = (best_rank == 17)
    return {
        "Stages (s)": s,
        "2s - 6 Parameters": num_params,
        "Order Conditions": 17,
        "Count Test OK?": (num_params >= 17),
        "Numeric Rank Test?": passed,
        "Numeric Rank": best_rank,
        "Structure": "2S−6 ESRK (Van der Houwen style)"
    }

# =============================================================================
# Example of how to run a small sweep:
# =============================================================================
if __name__ == "__main__":
    for s in range(9, 31):   # test s=9..30
        print(f"\nChecking s = {s} …")
        result = verify_2s_minus_6_matches_rooted_trees(s, nprobe=10)
        if result["Numeric Rank Test?"]:
            print(f"→ First valid s = {s} : {result}")
            break
        else:
            print(f"   failed: rank = {result['Numeric Rank']} / 17")
    else:
        print("No s ≤ 30 passed. Consider raising the upper bound.")
