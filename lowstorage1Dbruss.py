import numpy as np
import time
import tracemalloc
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Define the coefficients for the Runge-Kutta method
a= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0243586417803786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0258303808904268, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0667956303329210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0140960387721938, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0412105997557866, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0149469583607297, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.414086419082813, 0, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00395908281378477, 0, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.480561088337756, 0, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.319660987317690, 0, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0.00668808071535874, 0, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.0374638233561973, 0, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.439499983548480, 0, 0], [0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.0358989324994081, 0.00661245794721050, 0.216746869496930, 0, 0.422645975498266, 0.0327614907498598, 0.367805790222090, 0]]

b=[0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.035898932499408134, 0.006612457947210495, 0.21674686949693006, 0.0, 0.42264597549826616, 0.03276149074985981, 0.0330623263939421, 0.0009799086295048407]


a_values = np.array(a)

b_values = np.array(b)

# Kahan summation for improved floating-point precision
def kahan_sum(values):
    sum_ = np.float128(0.0)
    compensation = np.float128(0.0)
    for value in values:
        y = value - compensation
        temp = sum_ + y
        compensation = (temp - sum_) - y
        sum_ = temp
    return sum_

# 2D Brusselator ODE function
def brusselator_2d(t, y):
    x, y_ = y
    a, b = 1.0, 1.0  # Example parameters for the Brusselator
    dx_dt = a - (b + 1) * x + x**2 * y_
    dy_dt = b * x - x**2 * y_
    return np.array([dx_dt, dy_dt], dtype=np.float128)

# Runge-Kutta integration with optimized low storage (2 Registers)
def rk15_low_storage_2_registers_optimized(f, t_span, y0, h, a_values, b):
    t0, tf = t_span
    t = np.float128(t0)
    y = y0.copy()
    t_values = [t0]
    y_values = [y.copy()]
    
    s = len(b)
    num_steps = int(np.ceil((tf - t0) / h))
    c = np.array([kahan_sum(a_values[i][:i]) for i in range(s)], dtype=np.float128)
    
    # Determine the last usage of each k_j
    last_usage = [0] * s
    for j in range(s):
        for i in reversed(range(s)):
            if a_values[i][j] != 0:
                last_usage[j] = i
                break
    
    # Initialize two registers
    R1 = np.zeros_like(y, dtype=np.float128)
    R2 = np.zeros_like(y, dtype=np.float128)
    
    # Track which k_j is stored in which register
    register1_kj = None
    register2_kj = None
    
    # Temporary storage for k_j that cannot be stored in registers
    temp_k = {}
    print(temp_k)
    for step in range(num_steps):
        sum_bF = np.zeros_like(y, dtype=np.float128)
        
        for i in range(s):
            y_stage = y.copy()
            
            for j in range(i):
                # Retrieve k_j from registers or temporary storage
                if j in temp_k:
                    k_j = temp_k[j]
                elif j == register1_kj:
                    k_j = R1
                elif j == register2_kj:
                    k_j = R2
                else:
                    raise ValueError(f"k_{j+1} is needed but not stored.")
                
                y_stage += h * a_values[i][j] * k_j
            
            t_stage = t + c[i] * h
            k_new = f(t_stage, y_stage)
            
            # Determine which register can be overwritten
            overwrite_j = []
            if register1_kj is not None and last_usage[register1_kj] <= i:
                overwrite_j.append(register1_kj)
            if register2_kj is not None and last_usage[register2_kj] <= i:
                overwrite_j.append(register2_kj)
            
            if overwrite_j:
                # Assign k_new to the first available register that can be overwritten
                j_to_overwrite = overwrite_j[0]
                if j_to_overwrite == register1_kj:
                    R1 = k_new.copy()
                    register1_kj = i
                elif j_to_overwrite == register2_kj:
                    R2 = k_new.copy()
                    register2_kj = i
            else:
                # Assign k_new to a register if available
                if register1_kj is None:
                    R1 = k_new.copy()
                    register1_kj = i
                elif register2_kj is None:
                    R2 = k_new.copy()
                    register2_kj = i
                else:
                    # Both registers are occupied; store in temporary storage
                    temp_k[i] = k_new.copy()
            
            # Accumulate the weighted derivative
            sum_bF += b[i] * k_new
        
        # Update the solution
        y += h * sum_bF
        t += h
        t_values.append(t)
        y_values.append(y.copy())
        #print(temp_k)
        # Clear temporary storage for next step
        temp_k.clear()
    
    return np.array(t_values, dtype=np.float128), np.array(y_values, dtype=np.float128)

# L2 norm error calculation
def l2_norm_error(y_numerical, y_reference):
    error = y_numerical - y_reference
    l2_norm = np.linalg.norm(error, axis=1)
    return np.sqrt(np.mean(l2_norm**2, dtype=np.float128))

# Function to calculate order of convergence
def calculate_order_of_convergence(errors, hs):
    orders = []
    for i in range(1, len(errors)):
        if errors[i-1] == 0:
            orders.append(np.nan)
        else:
            order = np.log(errors[i] / errors[i-1]) / np.log(hs[i] / hs[i-1])
            orders.append(order)
    return orders

# Interpolate solution for error calculation
def interpolate_solution(t_values, y_values, t_values_ref):
    y_interpolated = []
    for i in range(y_values.shape[1]):
        f_interp = interp1d(t_values, y_values[:, i], kind='cubic', fill_value="extrapolate")
        y_interpolated.append(f_interp(t_values_ref))
    return np.array(y_interpolated).T

if __name__ == "__main__":
    # Parameters for convergence study
    t_span = (0, 10)  # Simulate over a longer time to observe Brusselator dynamics
    y0 = np.array([1.2, 2.5], dtype=np.float128)  # Initial conditions for Brusselator
    errors = []
    hs = np.linspace(0.05, 0.001, 50)  # Different time step sizes
    iterations_list = []
    memory_usage = []
    computation_times = []
    
    # Generate the reference solution with high precision using full storage
    print("Generating reference solution with h=0.001...")
    t_values_ref, y_values_ref = rk15_low_storage_2_registers_optimized(
        brusselator_2d, t_span, y0, 0.00001, a_values, b_values)
    print("Reference solution generated.")
    
    for h in hs:
        print(f"Running RK15 with h={h} using 2 optimized registers...")
        # Start memory tracking
        tracemalloc.start()
        
        # Start time measurement
        start_time = time.time()
        
        # Run numerical method
        t_values, y_values = rk15_low_storage_2_registers_optimized(
            brusselator_2d, t_span, y0, h, a_values, b_values)
        
        # Record computation time and memory usage
        computation_times.append(time.time() - start_time)
        current, peak = tracemalloc.get_traced_memory()
        memory_usage.append(peak)  # Peak memory usage
        tracemalloc.stop()
        
        # Interpolate solution
        y_values_interpolated = interpolate_solution(t_values, y_values, t_values_ref)
        
        # Calculate L2 norm error
        error = l2_norm_error(y_values_interpolated, y_values_ref)
        errors.append(error)
        iterations_list.append(len(t_values) - 1)  # Number of iterations
        
        print(f"h={h}: Error={error}, Iterations={len(t_values)-1}, Time={computation_times[-1]:.6f}s, Memory={peak} bytes")
    
    # Calculate order of convergence
    orders = calculate_order_of_convergence(errors, hs)
    
    # Print results
    print("\nRunge-Kutta 15 Method with Optimized Low Storage (2 Registers) on 1D Brusselator")
    print("Step sizes:", hs)
    print("Errors:", errors)
    print("Orders of convergence:", orders)
    print("Iterations per step size:", iterations_list)
    print("Computation times (s):", computation_times)
    print("Memory usage (bytes):", memory_usage)
    
    # Plotting the convergence
    plt.figure(figsize=(8,6))
    plt.loglog(hs, errors, 'o-', label='Numerical Error')
    plt.loglog(hs, [errors[0]*(h/hs[0])**3 for h in hs], 'k--', label='Third Order')
    plt.xlabel('Time Step Size (Δt)')
    plt.ylabel('Error Norm')
    plt.title('Convergence Study of 2D Brusselator')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
