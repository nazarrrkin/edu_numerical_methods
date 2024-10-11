import numpy as np

n = 5
entries= np.random.randint(low = -10, high = 10, size = n)
D = np.diag(np.random.rand(n))
C = np.random.rand(n,n)
A = np.linalg.inv(C) @ D @ C

def power_method(A, delta = 1e-8):
    x = np.random.rand(n)
    x /= np.linalg.norm(x)
    while True:
        Ax = np.dot(A, x)
        vector = Ax / np.linalg.norm(Ax)
        if np.linalg.norm(vector - x) < delta:
            break
        x = vector
    eigenvalue = np.dot(np.dot(x, A), x) / np.dot(x, x)
    return eigenvalue, x

eigenvalue, eigenvector = power_method(A)

#print("\n\t\tBiggest eigenvalue by pm:", eigenvalue)
#print("Appropriate eigenvector:", eigenvector)

def inverse_pm_with_shift(A, mu, n = n, tol = 1e-6, max_iter = 1000):
    E = np.eye(n)
    A_shifted = A - mu * E

    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    for i in range(max_iter):
        y = np.linalg.solve(A_shifted, x)
        x_new = y / np.linalg.norm(y)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    eigenvalue = np.dot(x.T, np.dot(A, x))
    return eigenvalue, x

def find_all_eigenpairs(A, shifts, n = n):
    eigenpairs = []

    for mu in shifts:
        eigenvalue, eigenvector = inverse_pm_with_shift(A, mu)
        eigenpairs.append((eigenvalue, eigenvector))

    return eigenpairs


shifts = np.linspace(0, 1, n)

eigenpairs = find_all_eigenpairs(A, shifts)

#for i, (eigenvalue, eigenvector) in enumerate(eigenpairs):
 #   print(f"\nEigenvalue by inverse pm {i+1}: {eigenvalue}")
  #  print(f"Eigenvector {i+1}: {eigenvector}")

def householder_reflection(A):
    n = A.shape[0]
    H = np.copy(A)
    Q = np.eye(n)

    for k in range(n-2):
        x = H[k+1:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)

        Q_k = np.eye(n)
        Q_k[k+1:, k+1:] -= 2.0 * np.outer(v, v)
        H = Q_k @ H @ Q_k
        Q = Q @ Q_k

    return Q, H

def qr_decomposition(H):
    n = H.shape[0]
    Q = np.eye(n)
    R = np.copy(H)

    for i in range(n-1):
        for j in range(i+1, n):
            if R[j, i] != 0:
                r = np.hypot(R[i, i], R[j, i])
                c = R[i, i] / r
                s = -R[j, i] / r
                G = np.eye(n)
                G[i, i] = c
                G[i, j] = -s
                G[j, i] = s
                G[j, j] = c

                R = G @ R
                Q = Q @ G.T

    return Q, R

def shift(H):
    n = H.shape[0]
    d = (H[n-2, n-2] - H[n-1, n-1]) / 2
    mu = H[n-1, n-1] - (np.sign(d) * H[n-1, n-2]**2) / (abs(d) + np.sqrt(d**2 + H[n-1, n-2]**2))
    return mu

def qr_algorithm(A, tol=1e-8, max_iter=1000):
    Q, H = householder_reflection(A)
    n = H.shape[0]
    eigenvalues = []

    for _ in range(max_iter):
        if n == 0:
            break

        mu = shift(H)
        I = np.eye(n)
        Q_k, R = qr_decomposition(H - mu * I)
        H = R @ Q_k + mu * I

        # Понижение размерности
        for i in range(n-1, 0, -1):
            if abs(H[i, i-1]) < tol:  # Проверка на малость элемента
                H[i, i-1] = 0
                eigenvalues.append(H[i, i])  # Сохранение найденного собственного значения
                H = H[:i, :i]  # Уменьшение размерности матрицы
                n = i  # Обновление размерности
                break

        off_diagonal = np.sum(np.abs(H[np.triu_indices(n, 1)]))
        if off_diagonal < tol:  # Проверка на сходимость
            eigenvalues.extend(np.diag(H))
            break

    return np.array(eigenvalues)

eigenvalues = qr_algorithm(A)
print("\nEigenvalues by QR:", eigenvalues)
