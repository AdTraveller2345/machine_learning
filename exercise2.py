import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    u, v = x
    return (u**2 + v - 11)**2 + (u + v**2 - 7)**2

def grad_f(x):
    """Gradient of the function f."""
    u, v = x
    df_du = 4 * u * (u**2 + v - 11) + 2 * (u + v**2 - 7)
    df_dv = 2 * (u**2 + v - 11) + 4 * v * (u + v**2 - 7)
    return [df_du, df_dv]

def gradient_descent(f, grad_f, eta, u0, v0, max_iter=100) -> tuple[list, list]:
    """Performs gradient descent."""
    u, v = u0, v0
    path = [(u, v)]
    values = [f((u, v))]
    
    for t in range(max_iter):
        grad = grad_f((u, v))
        step_size = eta(t)
        u -= step_size * grad[0]
        v -= step_size * grad[1]
        path.append((u, v))
        values.append(f((u, v)))
    
    return path, values

def eta_const(t,c=1e-3) -> float:
    return c

def eta_sqrt(t,c=1e-3) -> float:
    return c / math.sqrt(t + 1)

def eta_multistep(t, milestones=[30, 80, 100], c=1e-3, eta_init=1e-3) -> float:
    if t < milestones[0]: return eta_init
    elif t < milestones[1]: return eta_init*c
    else: return eta_init*c*c

# Run gradient descent with different step-size strategies
u0, v0 = 4, -5
max_iter = 100

# Constant step size
path_const, values_const = gradient_descent(f, grad_f, eta_const, u0, v0, max_iter)
print("Constant Step Size:")
print(f"f(u100, v100) = {values_const[-1]}")
print(f"min f(ut, vt) = {min(values_const)}")

# Decreasing step size
path_sqrt, values_sqrt = gradient_descent(f, grad_f, eta_sqrt, u0, v0, max_iter)
print("\nDecreasing Step Size:")
print(f"f(u100, v100) = {values_sqrt[-1]}")
print(f"min f(ut, vt) = {min(values_sqrt)}")

# Multi-step step size
path_multistep, values_multistep = gradient_descent(f, grad_f, eta_multistep, u0, v0, max_iter)
print("\nMulti-step Step Size:")
print(f"f(u100, v100) = {values_multistep[-1]}")
print(f"min f(ut, vt) = {min(values_multistep)}")

# Different starting points
starting_points = [(-4, 0), (0, 0), (4, 0), (0, 4), (5, 5)]

print("\nInitialization Results (Constant Step Size):")
for i, (u0, v0) in enumerate(starting_points, start=1):
    path, values = gradient_descent(f, grad_f, eta_const, u0, v0, max_iter)
    final_u, final_v = path[-1]
    final_f = values[-1]
    print(f"p{i} -> u: {final_u}, v: {final_v}, f_final: {final_f}")