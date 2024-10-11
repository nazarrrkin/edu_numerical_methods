import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
from scipy.interpolate import lagrange

def f(x):
    return x + np.log(x)

a, b = 1, 10  # Интервал
n_values = [5, 10, 15, 20]  # Разные количества узлов

x_dense = np.linspace(a, b, 1000)
y_true = f(x_dense)

def build_splines(n):
    x_nodes = np.linspace(a, b, n)
    y_nodes = f(x_nodes)

    S10 = interp1d(x_nodes, y_nodes, kind='linear')
    S21 = interp1d(x_nodes, y_nodes, kind='quadratic')
    S32 = CubicSpline(x_nodes, y_nodes)

    return x_nodes, y_nodes, S10, S21, S32

def plot_splines(n):
    x_nodes, y_nodes, S10, S21, S32 = build_splines(n)

    plt.figure(figsize=(12, 8))
    plt.plot(x_dense, y_true, label='True function', color='black')
    plt.plot(x_dense, S10(x_dense), label='S1,0 (Linear)')
    plt.plot(x_dense, S21(x_dense), label='S2,1 (Quadratic)')
    plt.plot(x_dense, S32(x_dense), label='S3,2 (Cubic)')
    plt.scatter(x_nodes, y_nodes, color='red')
    plt.legend()
    plt.title(f'Interpolation splines for n={n}')
    plt.show()

def max_deviation(n):
    x_nodes, y_nodes, S10, S21, S32 = build_splines(n)

    max_dev_S10 = np.max(np.abs(S10(x_dense) - y_true))
    max_dev_S21 = np.max(np.abs(S21(x_dense) - y_true))
    max_dev_S32 = np.max(np.abs(S32(x_dense) - y_true))

    return max_dev_S10, max_dev_S21, max_dev_S32

for n in n_values:
    max_dev_S10, max_dev_S21, max_dev_S32 = max_deviation(n)
    print(f'\n\tn={n}: Max deviation S1,0 = {max_dev_S10:.6f}, S2,1 = {max_dev_S21:.6f}, S3,2 = {max_dev_S32:.6f}')

for n in n_values:
    plot_splines(n)

def plot_error_distribution(n):
    x_nodes = np.linspace(a, b, n)
    y_nodes = f(x_nodes)

    S32 = CubicSpline(x_nodes, y_nodes)
    lagrange_poly = lagrange(x_nodes, y_nodes)

    error_S32 = np.abs(S32(x_dense) - y_true)
    error_lagrange = np.abs(lagrange_poly(x_dense) - y_true)

    plt.figure(figsize=(12, 8))
    plt.plot(x_dense, error_S32, label='Error S3,2 (Cubic)')
    plt.plot(x_dense, error_lagrange, label='Error Lagrange')
    plt.legend()
    plt.title(f'Error distribution for n={n}')
    plt.show()

for n in n_values:
    plot_error_distribution(n)
