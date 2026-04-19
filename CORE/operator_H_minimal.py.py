import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Загрузка данных
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
gram_indices = np.load('gram_indices_2M.npy')[:2_000_000]
gram_classes = gram_indices % 12

# Построение матрицы переходов из данных
M = np.zeros((12, 12))
for i in range(len(gram_indices) - 2):
    if gram_indices[i+1] - gram_indices[i] == 0:
        c1 = gram_classes[i]
        c3 = gram_classes[i+2]
        M[c1, c3] += 1

row_sums = M.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
M = M / row_sums

# Эрмитизация
H = (M + M.T) / 2

# Диагонализация
eigvals, _ = eigh(H)

# Сравнение с первыми нулями
first_zeros = zeros[:12]
corr = np.corrcoef(eigvals[:12], first_zeros)[0, 1]

print(f"Корреляция спектра H с нулями: {corr:.4f}")

# Визуализация
plt.figure(figsize=(10, 5))
plt.plot(eigvals[:12], 'o-', label='λ(H)', color='blue')
plt.plot(first_zeros / first_zeros.max() * eigvals[:12].max(), 's-', label='γ_n (норм)', color='red')
plt.legend()
plt.title(f'Корреляция = {corr:.4f}')
plt.grid()
plt.show()