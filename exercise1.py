import numpy as np

#part a

def argmin_x1(x):
    return (2 * x[2] + 3 * x[1] - 1) / 2

def argmin_x2(x):
    return (x[0] + 2 * x[2] + 5) / 6

def argmin_x3(x):
    return (x[0] + 3 * x[1] - 4) / 4

def partA(x):
    print(f"argmin x1 = {argmin_x1(x)}")
    print(f"argmin x2 = {argmin_x2(x)}")
    print(f"argmin x3 = {argmin_x3(x)}")

#output
partA([4,3,2])


#part b
def coordinate_descent(f, argmin, x_t0, max_iter=25):
    x = x_t0
    
    for t in range(max_iter):
        for i in range(len(x)):
            x[i] = argmin[i](x)
    
    return x

def f(x):
    x1, x2, x3 = x
    return np.exp(x1 - 3*x2 + 3) + np.exp(3*x2 - 2*x3 - 2) + np.exp(2*x3 - x1 + 2)

x_t0 = [1, 20, 5]
argmin_funcs = [argmin_x1, argmin_x2, argmin_x3]
final_x = coordinate_descent(f, argmin_funcs, x_t0)

# Output
for i,x in enumerate(final_x):
    print(f"final x{i} = {x}")
print("Function value at final x:", f(final_x))
