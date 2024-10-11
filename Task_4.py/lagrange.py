import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x + np.log(x)

a, b = 1, 10
node_sets = [1, 3, 10, 50, 100]

def optimal_nodes(a, b, n):
    x=[]
    for i in range(n):
        x.append(float(1/2 * ((b - a) * np.cos((2*i+1)*np.pi/(2*n+2)) + (b + a))))
    return x


def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    L = np.zeros_like(x)
    for i in range(n):
        li = np.ones_like(x)
        for j in range(n):
            if i != j:
                li *= (x - x_points[j]) / (x_points[i] - x_points[j])
        L += y_points[i] * li
    return L

def max_deviation(x_points, y_points, f, x):
    y_true = f(x)
    y_interpolated = lagrange_interpolation(x_points, y_points, x)
    deviations = np.abs(y_true - y_interpolated)
    return np.max(deviations)

def plot_interpolations(f, a, b, node_sets):
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)

    plt.figure(figsize=(10, 6))
    plt.plot(x_fine, y_fine, label='Original function', linewidth=2)

    for nodes in node_sets:
        x_points = np.linspace(a, b, nodes, dtype = float)
        #x_points = optimal_nodes(a, b, nodes)
        y_points = f(x_points)
        y_interpolated = lagrange_interpolation(x_points, y_points, x_fine)
        plt.plot(x_fine, y_interpolated, label=f'Interpolation with {nodes} nodes')

        max_dev = max_deviation(x_points, y_points, f, x_fine)
        print(f'\n\tMax deviation for {nodes} nodes: {max_dev}')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Lagrange Interpolation Polynomials')
    plt.grid(True)
    plt.show()

plot_interpolations(f, a, b, node_sets)
