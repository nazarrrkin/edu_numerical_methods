import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x + np.log(x)

a, b = 0.1, 1
node_sets = [1, 3, 10, 50]

def optimal_nodes(a, b, n):
    x=[]
    for i in range(n):
        x.append(float(1/2 * ((b - a) * np.cos((2*i+1)*np.pi/(2*n+2)) + (b + a))))
    return x

def divided_differences(x_points, y_points):
    n = len(x_points)
    coef = np.zeros([n, n])
    coef[:,0] = y_points

    for j in range(1, n):
        for i in range(n-j):
            coef[i,j] = (coef[i+1,j-1] - coef[i,j-1]) / (x_points[i+j] - x_points[i])

    return coef[0, :]
#пример построения ньютона сделать другой способ находить разделенные разности
#составить систему коэфов


def newton_polynomial(coef, x_points, x):
    n = len(x_points)
    result = coef[n-1]
    for k in range(1, n):
        result = coef[n-1-k] + (x - x_points[n-1-k]) * result
    return result

def newton_interpolation(x_points, y_points, x):
    coef = divided_differences(x_points, y_points)
    return newton_polynomial(coef, x_points, x)


def max_deviation(x_points, y_points, f, x):
    y_true = f(x)
    y_interpolated = np.array([newton_interpolation(x_points, y_points, xi) for xi in x])
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
        y_interpolated = np.array([newton_interpolation(x_points, y_points, xi) for xi in x_fine])
        plt.plot(x_fine, y_interpolated, label=f'Interpolation with {nodes} nodes')

        max_dev = max_deviation(x_points, y_points, f, x_fine)
        print(f'Max deviation for {nodes} nodes: {max_dev}')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Newton Interpolation Polynomials')
    plt.grid(True)
    plt.show()

plot_interpolations(f, a, b, node_sets)
