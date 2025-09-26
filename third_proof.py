

import sympy as sp
import random
from typing import Dict, List, Tuple, Sequence, Optional

# ---------------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------------

def build_params(S: int):
    """
    Allocate the (2S-8) symbols and slice them into:
      b_shared : single symbol used for stages 0..7 (if S>=8)
      b_unique : symbols for stages 8..S-1  (length S-8; may be 0 if S=8)
      d_diag   : subdiagonal symbols a_{i,i-1} for i=1..S-1 (length S-1)

    Returns dict with keys:
      params, b_shared, b_unique(list), d_diag(list)

    Raises AssertionError if S < 8.
    """
    assert S >= 8, "Need at least S = 8 stages (pattern assumes 8 shared b's)."
    n_params = 2*S - 8
    params   = sp.symbols(f'p0:{n_params}', real=True)

    b_shared = params[0]
    b_unique = params[1 : 1 + (S - 8)]         # may be empty list if S==8
    d_diag   = params[1 + (S - 8):]            # length S-1
    assert len(d_diag) == S-1

    return dict(
        params=params,
        b_shared=b_shared,
        b_unique=list(b_unique),
        d_diag=list(d_diag),
        n_params=n_params,
    )


def build_b_list(S: int, b_shared, b_unique: Sequence[sp.Symbol]):
    """
    Build the stage weight list b[0..S-1].
    First min(8,S) entries all refer to b_shared; after that use b_unique.
    """
    b = [b_shared]*min(8,S) + list(b_unique)
    return b[:S]


def build_a_dict(S: int, b: Sequence[sp.Symbol], d_diag: Sequence[sp.Symbol]):
    """
    Build strict-lower 'a' entries for the VdH tying pattern.

    Conventions:
      Rows 0..S-1, Cols 0..i-1.
      Subdiagonal (i,j=i-1) has its own symbol d_diag[i-1].
      All earlier columns j < i-1 reuse b[j].

    Returns dict keyed by (i,j) -> sympy expr.
    """
    assert len(d_diag) == S-1
    a = {}
    for i in range(S):
        for j in range(i):
            if j == i - 1:
                a[(i,j)] = d_diag[i-1]
            else:
                a[(i,j)] = b[j]
    return a


def build_c_list(S: int, a: Dict[Tuple[int,int], sp.Expr]):
    """
    Abscissae c_i = sum_{j<i} a_{i,j}.
    c_0 = 0 by definition.
    """
    c = []
    for i in range(S):
        if i == 0:
            c.append(sp.Integer(0))
        else:
            c.append(sp.simplify(sum(a[(i,j)] for j in range(i))))
    return c


# ---------------------------------------------------------------------------
# Order conditions
# ---------------------------------------------------------------------------

def order_conditions_order3(S: int,
                            b: Sequence[sp.Expr],
                            a: Dict[Tuple[int,int], sp.Expr],
                            c: Sequence[sp.Expr]):
    """
    Return the four 3rd-order rooted-tree order conditions (==0 when satisfied):

        OC1: sum b_i = 1
        OC2: sum b_i c_i = 1/2
        OC3: sum b_i c_i^2 = 1/3
        OC4: sum_i b_i sum_{j<i} a_{ij} c_j = 1/6

    Returns list [OC1,OC2,OC3,OC4].
    """
    OC1 = sp.simplify(sp.Add(*b) - sp.Integer(1))
    OC2 = sp.simplify(sp.Add(*[b[i]*c[i] for i in range(S)]) - sp.Rational(1,2))
    OC3 = sp.simplify(sp.Add(*[b[i]*c[i]**2 for i in range(S)]) - sp.Rational(1,3))
    OC4 = sp.Integer(0)
    for i in range(S):
        if i == 0:
            continue
        inner = sp.Add(*[a[(i,j)]*c[j] for j in range(i)])
        OC4 += b[i]*inner
    OC4 += b[0]*0  # explicit, for completeness
    OC4 = sp.simplify(OC4 - sp.Rational(1,6))
    return [OC1,OC2,OC3,OC4]


# ---------------------------------------------------------------------------
# Symbolic verification (no numeric solve)
# ---------------------------------------------------------------------------

def verify_two_register_order3_symbolic(S: int, verbose=False):
    """
    Build symbolic structure for given S and compute symbolic Jacobian rank.

    Returns (rank, dof, J, conds, symdict) where:
      rank   : symbolic rank of 4 x (2S-8) Jacobian
      dof    : (2S-8) - rank
      J      : sympy Matrix
      conds  : list of 4 sympy expressions (==0 when satisfied)
      symdict: dict returned by build_params (includes params, etc.)
    """
    syms = build_params(S)
    b = build_b_list(S, syms['b_shared'], syms['b_unique'])
    a = build_a_dict(S, b, syms['d_diag'])
    c = build_c_list(S, a)
    conds = order_conditions_order3(S, b, a, c)

    params = syms['params']
    J = sp.Matrix([[sp.diff(eq, p) for p in params] for eq in conds])
    rank = J.rank()
    dof  = syms['n_params'] - rank

    if verbose:
        print(f"Symbolic: S={S}, n_params={syms['n_params']}, rank={rank}, dof={dof}")

    return rank, dof, J, conds, syms


# ---------------------------------------------------------------------------
# Numeric solving helper
# ---------------------------------------------------------------------------

def _default_solve_syms(S: int, syms) -> List[sp.Symbol]:
    """
    Pick 4 parameters to solve for.
    Default: [b_shared, first b_unique (or first d if S==8), first two d_diag].
    """
    b_shared = syms['b_shared']
    b_unique = syms['b_unique']
    d_diag   = syms['d_diag']

    solve_syms = [b_shared]
    if S > 8:
        solve_syms.append(b_unique[0])
    else:
        solve_syms.append(d_diag[0])
    solve_syms.append(d_diag[0])
    if len(d_diag) > 1:
        solve_syms.append(d_diag[1])
    else:
        solve_syms.append(syms['params'][-1])
    return solve_syms[:4]


def verify_two_register_order3(S: int,
                               numeric: bool = True,
                               solve_syms: Optional[Sequence[sp.Symbol]] = None,
                               seed: int = 0,
                               max_tries: int = 5,
                               verbose: bool = False):
    """
    Full verification driver.

    Steps:
      1. Build symbolic structures (b,a,c,conds,J).
      2. Compute symbolic rank.
      3. If numeric=True:
           a. Choose 4 solve symbols (default _default_solve_syms).
           b. Randomly seed all params (uniform(-0.25,0.25)); hold all except solve_syms fixed.
           c. Use nsolve to satisfy 4 order conditions.
           d. Evaluate Jacobian numerically at solution; report numeric rank.

    Returns dict:
      {
        'S': S,
        'n_params': 2S-8,
        'symbolic_rank': int,
        'numeric_rank': int or None,
        'dof': n_params - 4,
        'params': tuple of params,
        'solve_syms': list of 4 syms used in numeric solve (if numeric),
        'solution': dict param->float (if numeric solution found),
        'Residuals': list of OC expressions,
        'Jacobian': sympy Matrix
      }
    """
    rank, dof, J, conds, syms = verify_two_register_order3_symbolic(S, verbose=verbose)

    # numeric attempt --------------------------------------------------------
    numeric_rank = None
    solution_subs = None
    solve_syms_used = None

    if numeric:
        rng = random.Random(seed)
        params = syms['params']
        # choose unknowns
        if solve_syms is None:
            solve_syms_used = _default_solve_syms(S, syms)
        else:
            solve_syms_used = list(solve_syms)

        # quick validity: need 4 distinct symbols
        if len(set(solve_syms_used)) != 4:
            raise ValueError("solve_syms must contain 4 distinct parameters.")

        # build substitution template
        # we will retry up to max_tries random seeds if nsolve has trouble
        for attempt in range(max_tries):
            subs_seed = {}
            for p in params:
                v = rng.uniform(-0.25, 0.25)
                if abs(v) < 1e-12:
                    v = 0.1
                subs_seed[p] = sp.nsimplify(v)

            fixed_subs = {p:subs_seed[p] for p in params if p not in solve_syms_used}
            R_num = [sp.simplify(eq.subs(fixed_subs)) for eq in conds]

            try:
                guess = [float(subs_seed[s]) for s in solve_syms_used]
                # high precision to help convergence
                sol = sp.nsolve([sp.N(expr) for expr in R_num], list(solve_syms_used), guess,
                                tol=1e-28, maxsteps=200, prec=80)
                # success
                solution_subs = dict(subs_seed)
                for s, val in zip(solve_syms_used, sol):
                    solution_subs[s] = float(val)

                # numeric Jacobian
                J_num = sp.Matrix([[sp.N(e.subs(solution_subs)) for e in row]
                                   for row in J.tolist()])
                numeric_rank = J_num.rank()
                if verbose:
                    print(f"nsolve succeeded on attempt {attempt+1}; numeric rank={numeric_rank}")
                break
            except Exception as exc:
                if verbose:
                    print(f"Attempt {attempt+1} failed: {exc}")
                continue

    out = {
        'S': S,
        'n_params': syms['n_params'],
        'symbolic_rank': int(rank),
        'numeric_rank': None if numeric_rank is None else int(numeric_rank),
        'dof': syms['n_params'] - 4,   # theoretical: params minus 4 OC
        'params': syms['params'],
        'solve_syms': solve_syms_used,
        'solution': solution_subs,
        'Residuals': conds,
        'Jacobian': J,
    }
    return out


# ---------------------------------------------------------------------------
# Jacobian minor utilities
# ---------------------------------------------------------------------------

def jacobian_minor_det(J: sp.Matrix,
                       params: Sequence[sp.Symbol],
                       cols: Sequence[sp.Symbol]):
    """
    Extract the 4x4 Jacobian minor using the param symbols in 'cols'
    (must be in 'params' and distinct), and return its determinant.
    """
    idxs = [params.index(c) for c in cols]
    M = J[:, idxs]
    detM = sp.simplify(M.det())
    return M, detM


# ---------------------------------------------------------------------------
# Demonstration / sweep
# ---------------------------------------------------------------------------

def demo_sweep(S_values: Sequence[int],
               numeric: bool = False,
               verbose: bool = False,
               seed_base: int = 0):
    """
    Run verify_two_register_order3 across a list of S and print summary table.
    """
    rows = []
    for k, S in enumerate(S_values):
        out = verify_two_register_order3(S,
                                         numeric=numeric,
                                         seed=seed_base + k,
                                         verbose=verbose)
        rows.append(out)

    # Pretty print summary
    print(f"{'S':>4} {'n_params':>8} {'sym_rank':>8} {'num_rank':>8} {'dof':>6}")
    for out in rows:
        print(f"{out['S']:>4} {out['n_params']:>8} {out['symbolic_rank']:>8} "
              f"{str(out['numeric_rank']):>8} {out['dof']:>6}")

    return rows


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # quick smoke tests
    for S in range(15,100,1):
        out = verify_two_register_order3(S, numeric=True, seed=42+S, verbose=True)
        print(f"S={S}: sym_rank={out['symbolic_rank']}, num_rank={out['numeric_rank']}")

    # example: compute a 4x4 minor determinant for S=15
    out15 = verify_two_register_order3(15, numeric=False)
    J15 = out15['Jacobian']
    params15 = out15['params']
    syms15 = build_params(15)
    cols = _default_solve_syms(15, syms15)
    M15, detM15 = jacobian_minor_det(J15, params15, cols)
    print("\nExample 4x4 minor determinant for S=15:")
    print(detM15)
