import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configure NumPy to raise exceptions on floating-point errors
np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

def kahan_sum(values):
    """
    Kahan summation for improved numerical accuracy.
    """
    sum_ = np.float128(0.0)
    compensation = np.float128(0.0)
    for value in values:
        y = np.float128(value) - compensation
        temp = sum_ + y
        compensation = (temp - sum_) - y
        sum_ = temp
    return sum_

def brusselator_1d(t, y, N, dx, D_u, D_v, a_param, b_param):
    """
    1D Brusselator reaction-diffusion system.
    """
    u = y[:N]
    v = y[N:]

    du_dt = np.zeros(N, dtype=np.float128)
    dv_dt = np.zeros(N, dtype=np.float128)
    u_xx = np.zeros(N, dtype=np.float128)
    v_xx = np.zeros(N, dtype=np.float128)

    # Internal points (finite difference)
    u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    v_xx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2

    # Neumann boundary conditions
    u_xx[0] = (u[1] - 2 * u[0] + u[2]) / dx**2
    u_xx[-1] = (u[-2] - 2 * u[-1] + u[-3]) / dx**2
    v_xx[0] = (v[1] - 2 * v[0] + v[2]) / dx**2
    v_xx[-1] = (v[-2] - 2 * v[-1] + v[-3]) / dx**2

    reaction_u = a_param - (b_param + 1) * u + (u**2) * v
    reaction_v = b_param * u - (u**2) * v

    du_dt = D_u * u_xx + reaction_u
    dv_dt = D_v * v_xx + reaction_v

    return np.concatenate([du_dt, dv_dt])

def runge_kutta_general(f, t_span, y0, h, A, b, args=()):
    """
    General s-stage explicit Runge-Kutta integrator using Butcher tableau (A, b).

    Parameters
    ----------
    f : callable
        The ODE right-hand side (RHS).
    t_span : tuple (t0, tf)
        Start and end times.
    y0 : ndarray
        Initial condition.
    h : float
        Time step size.
    A : ndarray of shape (s, s)
        The coefficient matrix of the Butcher tableau.
    b : ndarray of shape (s,)
        The weights for the final combination of stages.
    args : tuple
        Additional parameters to pass to f.

    Returns
    -------
    t_values : ndarray
        Array of time points.
    y_values : ndarray
        Array of solutions at each time point.
    """
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.copy()

    # Number of stages
    s = len(b)
    # c_i = sum of row i of A up to i-1 (standard definition for explicit RK)
    c = np.array([kahan_sum(A[i, :i]) for i in range(s)], dtype=np.float128)

    t_values = [t]
    y_values = [y.copy()]

    # Figure out how many steps are required
    # (or step less if it overshoots tf)
    num_steps = int(np.ceil((tf - t0) / h))

    for step in range(num_steps):
        # If very close to tf, adjust h to hit tf exactly
        if t + h > tf + np.float128(1e-14):
            h = tf - t
        if h <= 0:
            break

        # Store stage slopes here
        k_stages = np.zeros((s, y.size), dtype=np.float128)

        # --- Compute all s stages ---
        for i in range(s):
            # Construct the stage value y_stage
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * A[i, j] * k_stages[j]

            t_stage = t + c[i] * h
            k_stages[i] = f(t_stage, y_stage, *args)

        # --- Combine stages to get y_{n+1} ---
        for i in range(s):
            y += h * b[i] * k_stages[i]

        t += h
        t_values.append(t)
        y_values.append(y.copy())

        if t >= tf - np.float128(1e-14):
            # We have reached or exceeded the final time
            break

    return np.array(t_values, dtype=np.float128), np.array(y_values, dtype=np.float128)

def l2_norm_error(y_numerical, y_reference):
    """
    L2 norm of the difference between
    y_numerical and y_reference at final time.
    """
    error = y_numerical - y_reference
    l2_norm = np.linalg.norm(error)
    return np.sqrt(np.mean(l2_norm**2, dtype=np.float128))

def calculate_order_of_convergence(errors, hs):
    """
    Given a list of errors and corresponding step sizes,
    compute the observed order of convergence between consecutive steps.
    """
    orders = []
    for i in range(1, len(errors)):
        if errors[i-1] == 0 or hs[i-1] == hs[i]:
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

if __name__ == "__main__":
    # Problem setup
    t_span = (0, 10)
    L = 10.0
    N = 100
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    D_u = 0.1
    D_v = 0.05
    a_param = 1.0
    b_param = 3.0

    np.random.seed(0)
    u0 = a_param + 0.1 * np.random.rand(N)
    v0 = b_param / a_param + 0.1 * np.random.rand(N)
    y0 = np.concatenate([u0, v0]).astype(np.float128)

    # Stability limit for an explicit scheme ~ dx^2 / (2 * D_max)
    D_max = max(D_u, D_v)
    dt_max = dx**2 / (2 * D_max)
    print(f"Maximum allowable Δt for stability: {dt_max}")

    # Time-step array for convergence study
    hs = np.linspace(dt_max * 0.9, 0.001, 120)

    # Butcher tableau (A, b). Make sure shape matches s x s with s = len(b)

    a_values = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0162559490865921, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0998207004735601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.00837659391110831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.816464547912156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.0643329663939142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.236165334635088, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00122595190532598, 0, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.0223576010406567, 0, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0889134695375643, 0, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0874545207377521, 0.0368868956618399, 0, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0874545207377521, 0.243787757096315, 0.0745836439853446, 0, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0874545207377521, 0.243787757096315, 0.176532421863008, 0.787431349374128, 0, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0874545207377521, 0.243787757096315, 0.176532421863008, 0.0416960156051088, 0.000818162718803893, 0, 0],
    [0.0196784177818622, 0.0410628665737309, 0.0196412620004881, 0.0600714358887177, 0.142151154570198, 0.0408176286315510, 0.00165793539872671, 0.00450791958651599, 0.0874545207377521, 0.243787757096315, 0.176532421863008, 0.0416960156051088, 6.92385225011362e-5, 0.213693146719044, 0]
], dtype=np.float64)


    b_values = np.array([
    0.019678417781862172, 0.04106286657373094, 0.019641262000488085, 0.06007143588871774, 
    0.14215115457019759, 0.04081762863155103, 0.0016579353987267072, 0.004507919586515985, 
    0.08745452073775208, 0.24378775709631514, 0.17653242186300802, 0.04169601560510882, 
    6.923852250113618e-05, 0.09743567581622212, 0.023435749927302442
], dtype=np.float64)
    A_list = a_values
    b_list  = b_values

    # Pad A so it's exactly s x s if needed
    A_raw = np.array(A_list, dtype=np.float128)
    s = len(b_list)
    if A_raw.shape[0] < s or A_raw.shape[1] < s:
        A_tmp = np.zeros((s, s), dtype=np.float128)
        A_tmp[:A_raw.shape[0], :A_raw.shape[1]] = A_raw
        A_raw = A_tmp

    b_raw = np.array(b_list, dtype=np.float128)

    print(f"Adjusted A shape: {A_raw.shape}, b length: {len(b_raw)}")

    print("Generating reference solution with h=0.0001...")
    t_values_ref, y_values_ref = runge_kutta_general(
        brusselator_1d, t_span, y0, 0.0001, A_raw, b_raw,
        args=(N, dx, D_u, D_v, a_param, b_param)
    )
    print("Reference solution generated.")

    # -------------
    # Convergence study, with memory/time tracking
    # -------------

    # We'll store all relevant data here
    convergence_data = []

    for h in hs:
        # Track runtime
        start_time = time.time()

        # Track memory usage (start)
        tracemalloc.start()

        # Integrate
        t_values, y_values = runge_kutta_general(
            brusselator_1d, t_span, y0, h, A_raw, b_raw,
            args=(N, dx, D_u, D_v, a_param, b_param)
        )
        # Get memory usage (stop)
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        end_time = time.time()
        runtime = end_time - start_time

        # Compute error
        error = l2_norm_error(y_values[-1], y_values_ref[-1])

        # Number of timesteps actually performed
        num_steps = len(t_values) - 1

        # Store info in a dictionary
        convergence_data.append({
            "h": h,
            "error": error,
            "num_steps": num_steps,
            "runtime_s": runtime,
            "memory_current_kB": current_mem / 1024.0,
            "memory_peak_kB": peak_mem / 1024.0
        })

    # Extract arrays for final post-processing
    errors_array = np.array([d["error"] for d in convergence_data], dtype=float)
    hs_array = np.array([d["h"] for d in convergence_data], dtype=float)

    orders = calculate_order_of_convergence(errors_array, hs_array)

    # Print out the collected data
    print("\nConvergence Data:")
    for i, data in enumerate(convergence_data):
        h_ = data["h"]
        err_ = data["error"]
        num_ = data["num_steps"]
        mem_cur = data["memory_current_kB"]
        mem_peak = data["memory_peak_kB"]
        run_t = data["runtime_s"]
        # If we computed orders, be careful with indexing
        # We have len(orders) = len(errors)-1
        order_str = f"{orders[i-1]:.4f}" if (i > 0) else "N/A"
        print(f"h={h_:.6e}, error={err_:.6e}, order={order_str}, "
              f"steps={num_}, time={run_t:.4f}s, "
              f"memCurrent={mem_cur:.2f} kB, memPeak={mem_peak:.2f} kB")

    # Plotting
    plt.figure()
    plt.loglog(hs_array, errors_array, 'o-', label='Numerical Error')
    expected_order = 3
    plt.loglog(
        hs_array,
        [errors_array[0] * (h_ / hs_array[0])**expected_order for h_ in hs_array],
        'k--', label=f'{expected_order}rd-Order Reference'
    )
    plt.xlabel('Time Step Size (Δt)')
    plt.ylabel('Error Norm')
    plt.title('Convergence Study 1D Brusselator (3rd-Order, 15-stage)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
