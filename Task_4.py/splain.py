import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def f(x):
    return x + np.log(x)

a,b = 1, 10
nodes = np.linspace(a, b, 3)
values = f(nodes)

class CubicSpline:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        self.a, self.b, self.c, self.d = self.compute_coefficients()

    def compute_coefficients(self):
        h = np.diff(self.x)
        b = np.diff(self.y) / h
        u = np.zeros(self.n)
        v = np.zeros(self.n)
        u[1] = 2 * (h[0] + h[1])
        v[1] = 6 * (b[1] - b[0])
        for i in range(2, self.n - 1):
            u[i] = 2 * (h[i] + h[i - 1]) - h[i - 1] ** 2 / u[i - 1]
            v[i] = 6 * (b[i] - b[i - 1]) - h[i - 1] * v[i - 1] / u[i - 1]
        a = np.zeros(self.n)
        b = np.zeros(self.n)
        c = np.zeros(self.n)
        d = np.zeros(self.n)
        a[-1] = 0
        b[-1] = 0
        c[-1] = 0
        d[-1] = 0
        for i in range(self.n - 2, 0, -1):
            c[i] = (v[i] - h[i] * c[i + 1]) / u[i]
            b[i] = b[i + 1] - c[i + 1] * h[i]
            a[i] = (self.y[i + 1] - self.y[i]) / h[i] - h[i] * (b[i + 1] + 2 * c[i + 1]) / 6
            d[i] = (b[i + 1] - b[i]) / (6 * h[i])
        return a, b, c, d

    def __call__(self, z):
        idx = np.searchsorted(self.x, z)
        idx = np.array(idx, ndmin=1)
        idx[idx == 0] = 1
        idx[idx == len(self.x)] = len(self.x) - 1
        i = idx - 1
        h = z - self.x[i]
        return self.a[i] + self.b[i] * h + self.c[i] * h ** 2 / 2 + self.d[i] * h ** 3 / 6

spline_cubic = CubicSpline(nodes, values)

class LinearSpline:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, z):
        idx = np.searchsorted(self.x, z)
        idx = np.array(idx, ndmin=1)
        idx[(idx == 0).any()] = 1
        idx[(idx == len(self.x)).any()] = len(self.x) - 1
        i = idx - 1
        slope = (self.y[idx] - self.y[i]) / (self.x[idx] - self.x[i])
        return self.y[i] + slope * (z - self.x[i])

spline_linear = LinearSpline(nodes, values)

class QuadraticSpline:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, z):
        idx = np.searchsorted(self.x, z)
        idx = np.array(idx, ndmin=1)
        idx[(idx == 0).any()] = 1
        idx[(idx == len(self.x)).any()] = len(self.x) - 1
        i = idx - 1
        a = (self.y[i+1] - self.y[i]) / ((self.x[i+1] - self.x[i]) ** 2)
        b = (self.y[i] - a * (self.x[i] ** 2))
        return a * (z ** 2) + b * z

spline_quadratic = QuadraticSpline(nodes, values)

def max_deviation_spline(spline, f, a, b, n, m):
    x_values = np.linspace(a, b, m)
    f_values = [f(x) for x in x_values]
    spline_values = [spline(x) for x in x_values]

    deviations = [abs(spline_values[i] - f_values[i]) for i in range(m)]
    max_deviation = max(deviations)

    print('При n =',n , 'и m =', m, 'Отклонение сплайна =', max_deviation)

print('S10')
max_deviation_spline(spline_linear, f, a, b, 1, 100)
max_deviation_spline(spline_linear, f, a, b, 2, 100)
max_deviation_spline(spline_linear, f, a, b, 5, 100)
print('S21')
max_deviation_spline(spline_quadratic, f, a, b, 1, 100)
max_deviation_spline(spline_quadratic, f, a, b, 2, 100)
max_deviation_spline(spline_quadratic, f, a, b, 5, 100)
print('S32')
max_deviation_spline(spline_cubic, f, a, b, 1, 100)
max_deviation_spline(spline_cubic, f, a, b, 2, 100)
max_deviation_spline(spline_cubic, f, a, b, 5, 100)

x_values = np.linspace(1, 10, 100)
plt.plot(x_values, spline_linear(x_values), label='S10(x)')
plt.plot(x_values, spline_quadratic(x_values), label='S21(x)')
plt.plot(x_values, spline_cubic(x_values), label='S32(x)')
plt.plot(x_values, f(x_values), label='f(x)', linestyle='--')
plt.scatter(nodes, values, color='red', label='узлы')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Сплайны')
plt.legend()
plt.show()
