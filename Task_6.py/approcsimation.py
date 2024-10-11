import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x):
    return x * np.sqrt(x + 2)

def generate_data(f, x_values, error_range):
    data = []
    for x in x_values:
        f_values = []
        for i in range(3):
            f_values.append(f(x) + np.random.uniform(-error_range, error_range))
        data.append((x, f_values))
    return data

m = 51
x_values = np.linspace(-1, 1, m, dtype=float)
n_values = [1, 2, 3]
error_range = 0.5
data = generate_data(f, x_values, error_range)
f = np.array([np.mean(point[1]) for point in data])


def normal_equations(x, f, n):
    A = np.vander(x, n + 1, increasing=True)
    coeff = np.linalg.inv(A.T @ A) @ A.T @ f
    return coeff

def orthogonal_polinomials(n):
    coeff = []
    for i in range(n+1):
        devisible = 0
        devider = 0
        for j in range(len(x_values)):
            devisible += chebyshev(i, x_values[j]) * f[j]
            devider += pow(chebyshev(i, x_values[j]), 2)
        coeff.append(devisible / devider)
    return coeff


def chebyshev(j, x):
    if j == 0:
        return 1
    if j == 1:
        return x - 1/m * (sum(x_values))
    else:
        return x * chebyshev(j-1, x) - alpha(j) * x * chebyshev(j-1, x) - betta(j-1) * chebyshev(j-2, x)

def alpha(j):
    devisible = 0
    devider = 0
    for i in x_values:
        devisible += i * pow(chebyshev(j-1, i), 2)
        devider += pow(chebyshev(j-1, i), 2)
    return devisible / devider

def betta(j):
    devisible = 0
    devider = 0
    for i in x_values:
        devisible += i * chebyshev(j, i) * chebyshev(j-1, i)
        devider += pow(chebyshev(j-1, i), 2)
    return devisible / devider

def approcsimation_normal():
    plt.figure(figsize=(15, 10))
    for i, n in enumerate(n_values, start=1):
        plt.subplot(2, 3, i)

        x = np.array([point[0] for point in data])
        y_m = np.array([np.mean(point[1]) for point in data])
        y = np.array([point[1] for point in data])

        coeffs = normal_equations(x, y_m, n)
        fitted_curve = np.poly1d(coeffs[::-1])
        x_fit = np.linspace(min(x), max(x), 51)
        plt.plot(x, fitted_curve(x), label=f'Полином {n} степени', color='r')

        plt.xticks(np.linspace(min(x), max(x), num=10))
        plt.yticks(np.linspace(min(y_m), max(y_m), num=10))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Аппроксимация полиномом {n} степени')
        plt.legend()
        y0, y1, y2 = [], [], []
        for j in range(len(y_m)):
            y0.append(y[j][0])

        for d in range(len(y_m)):
            y1.append(y[d][1])

        for s in range(len(y_m)):
            y2.append(y[s][2])

        plt.scatter(x, y0, color='b', marker='o')
        plt.scatter(x, y1, color='b', marker='o')
        plt.scatter(x, y2, color='b', marker='o')

    plt.tight_layout()
    plt.show()

def approcsimation_ortogonal():
    plt.figure(figsize=(15, 10))
    for i, n in enumerate(n_values, start=1):
        plt.subplot(2, 3, i)

        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        y_m = np.array([np.mean(point[1]) for point in data])

        coeffs = orthogonal_polinomials(n)
        fitted_curve = np.poly1d(coeffs[::-1])

        plt.plot(x, fitted_curve(x), label=f'Полином {n} степени', color='r')

        plt.xticks(np.linspace(min(x), max(x), num=10))
        plt.yticks(np.linspace(min(y_m), max(y_m), num=10))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Аппроксимация полиномом {n} степени')
        plt.legend()
        y0, y1, y2 = [], [], []
        for j in range(len(y_m)):
            y0.append(y[j][0])

        for d in range(len(y_m)):
            y1.append(y[d][1])

        for s in range(len(y_m)):
            y2.append(y[s][2])

        plt.scatter(x, y0, color='b', marker='o')
        plt.scatter(x, y1, color='b', marker='o')
        plt.scatter(x, y2, color='b', marker='o')

    plt.tight_layout()
    plt.show()

def normal_squared_errors():
    for n in range(1, 6):
        coef = normal_equations(x_values,f,n)
        approx_polynomial = np.poly1d(coef[::-1])
        approx_y_values = approx_polynomial(x_values)
        errors = f  - approx_y_values
        sum_squared_errors = np.sum(errors ** 2)
        print(f'{sum_squared_errors} - sum of squared errors by {n}-degree polynomial from normal equations')


def orthogonal_squared_errors():
    for n in range(1, 6):
        coef = orthogonal_polinomials(n)
        approx_polynomial = np.poly1d(coef[::-1])
        approx_y_values = approx_polynomial(x_values)
        errors = f - approx_y_values
        sum_squared_errors = np.sum(errors ** 2)
        print(f'{sum_squared_errors} - sum of squared errors by {n}-degree orthogonal polynomial')


#approcsimation_ortogonal()
