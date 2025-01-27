import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configure NumPy to raise exceptions on floating-point errors
np.seterr(over='raise', invalid='raise', divide='raise', under='warn')

###############################################################################
# Kahan summation for improved floating-point precision
###############################################################################
def kahan_sum(values):
    """
    Perform Kahan summation on an array of values to reduce numerical error.
    """
    sum_ = np.float128(0.0)
    compensation = np.float128(0.0)
    for value in values:
        y = np.float128(value) - compensation
        temp = sum_ + y
        compensation = (temp - sum_) - y
        sum_ = temp
    return sum_

###############################################################################
# ODE Brusselator system
###############################################################################
def brusselator_ode(t, y, a_param, b_param):
    """
    The 2D (2-variable) Brusselator ODE system:
       du/dt = a - (b+1)*u + u^2 * v
       dv/dt = b*u - u^2 * v
    """
    u, v = y
    du_dt = a_param - (b_param + 1.0) * u + u**2 * v
    dv_dt = b_param * u - u**2 * v
    return np.array([du_dt, dv_dt], dtype=np.float128)

###############################################################################
# General Runge-Kutta integration function (16-stage ESRK)
###############################################################################
def runge_kutta_general(f, t_span, y0, h, a_values, b_values, args=()):
    """
    General Runge-Kutta integration with user-provided Butcher coefficients.

    Parameters:
    -----------
    f : function(t, y, *args) -> np.array
        The ODE system function.
    t_span : tuple (t0, tf)
        Start and end times.
    y0 : np.array
        Initial conditions.
    h : float
        Step size.
    a_values : np.ndarray
        Matrix of Runge-Kutta 'a' coefficients (s x s).
    b_values : np.ndarray
        Vector of Runge-Kutta 'b' coefficients (length s).
    args : tuple
        Additional arguments passed to the ODE function f.

    Returns:
    --------
    t_values : np.ndarray
        Time points at which the solution was recorded.
    y_values : np.ndarray
        Corresponding solutions at each time point.
    """
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.astype(np.float128)
    t_values = [t0]
    y_values = [y.copy()]

    s = len(b_values)  # Number of stages
    num_steps = int(np.ceil((tf - t0) / h))
    
    # Precompute c-values from a-values (sum of each row up to stage index)
    c_values = np.array([kahan_sum(a_values[i][:i]) for i in range(s)],
                        dtype=np.float128)

    for step in range(num_steps):
        # Adjust the last step to end exactly at tf
        if t + h > tf + np.float128(1e-12):
            h = tf - t

        # If h is zero or negative due to floating-point rounding, break
        if h <= 0:
            break

        k = [np.zeros_like(y, dtype=np.float128) for _ in range(s)]
        for i in range(s):
            # Compute intermediate stage
            y_stage = y.copy()
            for j in range(i):
                y_stage += h * a_values[i][j] * k[j]
            t_stage = t + c_values[i] * h

            try:
                k[i] = f(t_stage, y_stage, *args)
            except FloatingPointError as e:
                print(f"Floating point error at time {t_stage}: {e}")
                return (np.array(t_values, dtype=np.float128),
                        np.array(y_values, dtype=np.float128))

        # Combine stages to get the next solution
        for i in range(s):
            y += h * b_values[i] * k[i]

        t += h
        
        # Store results (avoid duplicates if t didn't advance numerically)
        if t > t_values[-1] + np.float128(1e-12):
            t_values.append(t)
            y_values.append(y.copy())

    return (np.array(t_values, dtype=np.float128),
            np.array(y_values, dtype=np.float128))

###############################################################################
# L2 norm error calculation
###############################################################################
def l2_norm_error(y_numerical, y_reference):
    """
    Calculate the L2 norm error between the numerical and reference solutions.
    y_numerical and y_reference are of shape (num_times, dimension).
    """
    error = y_numerical - y_reference
    # Norm along the 'dimension' axis, then root-mean-square over time
    l2_per_time = np.sqrt(np.sum(error**2, axis=1))
    return np.sqrt(np.mean(l2_per_time**2, dtype=np.float128))

###############################################################################
# Function to calculate order of convergence
###############################################################################
def calculate_order_of_convergence(errors, hs):
    """
    Calculate the experimental order of convergence given arrays of errors and
    corresponding step sizes hs.  The order between consecutive pairs of points:
        order = log(error[i] / error[i-1]) / log(hs[i] / hs[i-1])
    """
    orders = []
    for i in range(1, len(errors)):
        if errors[i-1] == 0 or hs[i-1] == hs[i]:
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

###############################################################################
# Interpolate solution for error calculation
###############################################################################
def interpolate_solution(t_values, y_values, t_values_ref):
    """
    Interpolate the solution y_values(t_values) to the reference time points
    t_values_ref, for each dimension in y_values.
    """
    dim = y_values.shape[1]
    y_interpolated = []
    for i in range(dim):
        f_interp = interp1d(t_values, y_values[:, i],
                            kind='cubic', fill_value="extrapolate")
        y_interpolated.append(f_interp(t_values_ref))
    return np.array(y_interpolated).T  # shape (len(t_values_ref), dim)

###############################################################################
# Main script
###############################################################################
if __name__ == "__main__":
    # Parameters for the Brusselator ODE
    a_param = 1.0
    b_param = 3.0

    # Time span for the simulation
    t_span = (0.0, 10.0)

    # Initial conditions (slightly perturbed around the steady state)
    # Steady state for Brusselator is (u, v) = (a_param, b_param/a_param).
    np.random.seed(0)
    u0 = a_param + 0.1 * np.random.rand()
    v0 = (b_param / a_param) + 0.1 * np.random.rand()
    y0 = np.array([u0, v0], dtype=np.float128)

    # A small helper for step sizes to test
    # (Feel free to adjust the range or distribution of hs as needed)
    hs = np.logspace(-3, -1, 100)  # e.g., from 1e-3 to 1e-1

    # Define the coefficients for the 16-stage ESRK method (fourth-order)
    # Please double-check these if needed. change to correct tableau


    ############################################################################
    # Generate the reference solution using a very small step size (e.g., 1e-5)
    ############################################################################
    print("Generating reference solution with h = 1e-5...")
    h_ref = 1e-5
    t_values_ref, y_values_ref = runge_kutta_general(
        brusselator_ode, t_span, y0, h_ref, a_values, b_values,
        args=(a_param, b_param)
    )
    print("Reference solution generated.")

    ############################################################################
    # Convergence study for different step sizes
    ############################################################################
    errors = []
    iterations_list = []
    memory_usage = []
    computation_times = []

    for h in hs:
        print(f"\nRunning ESRK method with h = {h} ...")
        # Start memory tracking
        tracemalloc.start()
        # Start time measurement
        start_time = time.time()

        # Run numerical method
        try:
            t_values, y_values = runge_kutta_general(
                brusselator_ode, t_span, y0, h, a_values, b_values,
                args=(a_param, b_param)
            )
        except FloatingPointError as e:
            print(f"Floating point error encountered at h = {h}: {e}")
            errors.append(np.nan)
            iterations_list.append(np.nan)
            computation_times.append(np.nan)
            memory_usage.append(np.nan)
            continue

        # Record computation time and memory usage
        comp_time = time.time() - start_time
        computation_times.append(comp_time)
        current, peak = tracemalloc.get_traced_memory()
        memory_usage.append(peak)  # Peak memory usage
        tracemalloc.stop()

        # Interpolate solution to the reference time grid
        y_values_interpolated = interpolate_solution(t_values, y_values, t_values_ref)

        # Calculate L2 norm error
        error = l2_norm_error(y_values_interpolated, y_values_ref)
        errors.append(error)
        iterations_list.append(len(t_values) - 1)  # Number of steps taken

        print(f"h={h}: Error={error:.6e}, Steps={len(t_values)-1}, "
              f"Time={comp_time:.6f}s, Memory={peak} bytes")

    # Calculate order of convergence
    orders = calculate_order_of_convergence(errors, hs)

    ############################################################################
    # Print final results
    ############################################################################
    print("\nBrusselator ODE Simulation with 16-Stage Fourth-Order ESRK Method")
    print("-----------------------------------------------------------------")
    print("Step sizes (h):", hs)
    print("Errors:", errors)
    print("Orders of convergence (pairwise):", orders)
    print("Iterations (number of steps) per h:", iterations_list)
    print("Computation times (s):", computation_times)
    print("Peak memory usage (bytes):", memory_usage)

    ############################################################################
    # Plot the convergence
    ############################################################################
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errors, 'o-', label='Numerical Error')
    
    # Reference line for expected order ~4
    expected_order = 4
    # We'll scale a reference line so it passes through the first data point
    # and has slope 'expected_order' in log-log scale.
    if len(errors) > 0 and errors[0] > 0:
        cst = errors[0] / (hs[0]**expected_order)
        plt.loglog(hs, [cst * (h**expected_order) for h in hs],
                   'k--', label=f'{expected_order}th Order Reference')

    plt.xlabel('Time Step Size (Δt)')
    plt.ylabel('L2 Error Norm')
    plt.title('Convergence Study of Brusselator ODE using 16-Stage ESRK (Fourth-Order)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
