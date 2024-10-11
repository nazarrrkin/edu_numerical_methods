from sympy import *
import numpy as np

def f(x):
    return x + np.log(x)

a, b = 1, 10
n = 5

def optimal_nodes(a, b, n):
    x=[]
    for i in range(n):
        x.append(float(1/2 * ((b - a) * cos((2*i+1)*np.pi/(2*n+2)) + (b + a))))
    return x

x_eq = np.linspace(a, b, n, dtype = float)
x_opt = optimal_nodes(a, b, n)

def Lagrange_equally(x, n):
    y = np.array(f(x))
    Pn = 0
    X = Symbol('X')
    for j in range(n):
        P = 1
        for i in range(n):
            if i == j: 
                continue
            P *= (X - x[i]) / (x[j] - x[i])
        P *= y[j]
        Pn += P
    return simplify(Pn)

print(Lagrange_equally(x_eq,n))

def divided_difference(x, y, k):
    if k == 0:
        return y[0]
    sum = 0
    for i in range(k + 1):
        product = y[i]
        for j in range(k + 1):
            if j != i:
                product /= (x[i] - x[j])
        sum += product
    return sum


def newton_interpolation_equal(x, y, xi):
    result = y[0]
    for i in range(1, len(x)):
        term = 1
        for j in range(i):
            term *= (xi - x[j])
        result += divided_difference(x, y, i) * term
    return result


def cubic_spline_interpolation(x, y):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1]

    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n-1):
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

    l[-1] = 1
    z[-1] = 0

    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)

    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])

    return y, b, c, d

def interpolate_spline(x, a, b, c, d, x_val):
    i = np.searchsorted(x, x_val) - 1
    if i == len(x) - 1:
        i -= 1
    h = x[i+1] - x[i]
    t = (x_val - x[i]) / h
    result = a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3
    return result

a, b, c, d = cubic_spline_interpolation(x_eq, np.array(f(x_eq)))
x_val =  2.5

spline_value = interpolate_spline(x_eq, a, b, c, d, x_val)
