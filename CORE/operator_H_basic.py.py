"""
ПОСТРОЕНИЕ ЭРМИТОВА ОПЕРАТОРА ИЗ МАТРИЦЫ ПРЫЖКОВ
Сравнение спектра с нулями дзета-функции
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
gram_indices = np.load('gram_indices_2M.npy')[:2_000_000]

gram_classes = gram_indices % 12

print(f"✓ Загружено {len(zeros):,} нулей")
print(f"✓ Диапазон высот: {zeros[0]:.2f} ... {zeros[-1]:.2f}")

# ============================================================
# 2. ПОСТРОЕНИЕ МАТРИЦЫ ПРЫЖКОВ (как в твоём анализе)
# ============================================================
print("\n" + "=" * 70)
print("ПОСТРОЕНИЕ МАТРИЦЫ ПРЫЖКОВ M[12×12]")
print("=" * 70)

# Ищем тройки (застревание → прыжок)
jump_matrix_counts = np.zeros((12, 12), dtype=int)

for i in range(len(gram_indices) - 2):
    diff1 = gram_indices[i+1] - gram_indices[i]
    
    if diff1 == 0:  # Нашли застревание
        c1 = gram_classes[i]
        c3 = gram_classes[i+2]
        jump_matrix_counts[c1, c3] += 1

# Нормируем по строкам (вероятности)
row_sums = jump_matrix_counts.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
M = jump_matrix_counts / row_sums

print(f"✓ Матрица M построена")
print(f"  Суммы по строкам: {M.sum(axis=1)}")

# ============================================================
# 3. ЭРМИТИЗАЦИЯ (симметризация)
# ============================================================
print("\n" + "=" * 70)
print("ЭРМИТИЗАЦИЯ: H = (M + Mᵀ)/2")
print("=" * 70)

H = (M + M.T) / 2

# Проверка на симметричность
is_symmetric = np.allclose(H, H.T)
print(f"✓ H симметрична (эрмитова): {is_symmetric}")

# ============================================================
# 4. ДИАГОНАЛИЗАЦИЯ
# ============================================================
print("\n" + "=" * 70)
print("ДИАГОНАЛИЗАЦИЯ H")
print("=" * 70)

eigenvalues, eigenvectors = eigh(H)  # eigh для симметричных матриц

# Сортируем по возрастанию
idx = np.argsort(eigenvalues)
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nСОБСТВЕННЫЕ ЗНАЧЕНИЯ H (эрмитова оператора):")
print("-" * 50)
for i, λ in enumerate(eigenvalues):
    print(f"  λ_{i:2d} = {λ:.6f}")

# ============================================================
# 5. СРАВНЕНИЕ С НУЛЯМИ ДЗЕТА-ФУНКЦИИ
# ============================================================
print("\n" + "=" * 70)
print("СРАВНЕНИЕ СОБСТВЕННЫХ ЗНАЧЕНИЙ С НУЛЯМИ")
print("=" * 70)

# Берём первые 12 нулей для сравнения
first_zeros = zeros[:12]

print(f"\n{'n':<4} {'λ_n (оператор)':<18} {'γ_n (нуль)':<18} {'Разница':<12}")
print("-" * 60)

for i in range(min(12, len(eigenvalues))):
    diff = abs(eigenvalues[i] - first_zeros[i])
    print(f"{i+1:<4} {eigenvalues[i]:<18.6f} {first_zeros[i]:<18.6f} {diff:<12.6f}")

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# График 1: Тепловая карта H
ax1 = axes[0]
im = ax1.imshow(H, cmap='RdBu_r', vmin=0, vmax=0.6)
ax1.set_xticks(range(12))
ax1.set_yticks(range(12))
ax1.set_xlabel('Конечный класс')
ax1.set_ylabel('Начальный класс')
ax1.set_title('Эрмитов оператор H = (M + Mᵀ)/2')
plt.colorbar(im, ax=ax1)

# График 2: Сравнение спектров
ax2 = axes[1]
n_range = np.arange(1, min(13, len(eigenvalues) + 1))
ax2.plot(n_range, eigenvalues[:12], 'o-', label='λ (оператор)', color='blue', markersize=8)
ax2.plot(n_range, first_zeros[:12], 's-', label='γ (нули ζ)', color='red', markersize=8)
ax2.set_xlabel('n')
ax2.set_ylabel('Значение')
ax2.set_title('Сравнение спектров (первые 12)')
ax2.legend()
ax2.grid(alpha=0.3)

# График 3: Разница
ax3 = axes[2]
diffs = [abs(eigenvalues[i] - first_zeros[i]) for i in range(min(12, len(eigenvalues)))]
ax3.bar(n_range, diffs, color='purple', alpha=0.7)
ax3.set_xlabel('n')
ax3.set_ylabel('|λ - γ|')
ax3.set_title('Абсолютная разница')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('hermitian_operator_spectrum.png', dpi=150)
print("\n✓ График сохранён как 'hermitian_operator_spectrum.png'")

# ============================================================
# 7. КОРРЕЛЯЦИЯ МЕЖДУ СПЕКТРАМИ
# ============================================================
print("\n" + "=" * 70)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 70)

# Нормализуем оба спектра для корреляции
norm_eigen = eigenvalues[:12] / eigenvalues[:12].max()
norm_zeros = first_zeros / first_zeros.max()

corr = np.corrcoef(norm_eigen, norm_zeros)[0, 1]
print(f"\nКорреляция Пирсона между спектрами: {corr:.6f}")

if abs(corr - 1.0) < 0.1:
    print("✅ БЛИЗКО! Спектр оператора коррелирует с нулями!")
elif abs(corr - 1.0) < 0.3:
    print("⚠️ Умеренная корреляция. Нужно уточнять оператор.")
else:
    print("❌ Корреляция слабая. Оператор требует доработки.")

# ============================================================
# 8. ВЫВОД
# ============================================================
print("\n" + "=" * 70)
print("ВЫВОД")
print("=" * 70)

print(f"""
Построен эрмитов оператор H = (M + Mᵀ)/2 на основе 12-потоковой структуры.

Собственные значения H:
{np.array2string(eigenvalues[:12], precision=6, suppress_small=True)}

Первые 12 нулей ζ(s):
{np.array2string(first_zeros, precision=6, suppress_small=True)}

Корреляция: {corr:.4f}

Если корреляция высокая (>0.9) — мы на правильном пути.
""")

plt.show()


# Используем ТОЛЬКО первые 2 точки для определения a и b
# Остальные 10 — для проверки

a_est = (first_zeros[1] - first_zeros[0]) / (eigenvalues[1] - eigenvalues[0])
b_est = first_zeros[0] - a_est * eigenvalues[0]

λ_pred = a_est * eigenvalues[:12] + b_est

# Сравниваем предсказания с реальными нулями (кроме первых 2)
for i in range(2, 12):
    print(f"γ_{i+1} реальный: {first_zeros[i]:.3f}, предсказанный: {λ_pred[i]:.3f}, ошибка: {abs(first_zeros[i] - λ_pred[i]):.3f}")


    import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Данные
eigenvals = np.array([-0.496084, -0.492805, -0.415838, -0.397435, -0.304218, -0.301957, 
                      -0.179609, 0.088479, 0.090574, 0.714885, 0.715181, 1.000035])
zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
                  40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248])

# Монотонное преобразование: γ = a * exp(b * λ) + c
def model(λ, a, b, c):
    return a * np.exp(b * λ) + c

# Обучаем на ПЕРВЫХ 6 точках
train_idx = range(6)
test_idx = range(6, 12)

popt, _ = curve_fit(model, eigenvals[train_idx], zeros[train_idx], 
                    p0=[50, 2, 10], maxfev=5000)

# Предсказание
γ_pred_train = model(eigenvals[train_idx], *popt)
γ_pred_test = model(eigenvals[test_idx], *popt)

print("ПАРАМЕТРЫ (ОПРЕДЕЛЕНЫ ПО 6 ТОЧКАМ):")
print(f"  a = {popt[0]:.3f}")
print(f"  b = {popt[1]:.3f}")
print(f"  c = {popt[2]:.3f}")

print("\nПРОВЕРКА НА ОБУЧАЮЩИХ (первые 6):")
for i in train_idx:
    print(f"  γ_{i+1}: реальный {zeros[i]:.3f}, предск. {γ_pred_train[i]:.3f}, ошибка {abs(zeros[i]-γ_pred_train[i]):.3f}")

print("\nЧЕСТНАЯ ПРОВЕРКА НА НОВЫХ (6 последних):")
errors = []
for i in test_idx:
    err = abs(zeros[i] - γ_pred_test[i - 6])
    errors.append(err)
    print(f"  γ_{i+1}: реальный {zeros[i]:.3f}, предск. {γ_pred_test[i-6]:.3f}, ошибка {err:.3f}")

print(f"\nСРЕДНЯЯ ОШИБКА на новых: {np.mean(errors):.3f}")
print(f"КОРРЕЛЯЦИЯ на новых: {np.corrcoef(zeros[test_idx], γ_pred_test)[0,1]:.4f}")

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Данные
eigenvals = np.array([-0.496084, -0.492805, -0.415838, -0.397435, -0.304218, -0.301957, 
                      -0.179609, 0.088479, 0.090574, 0.714885, 0.715181, 1.000035])
zeros = np.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178,
                  40.918719, 43.327073, 48.005151, 49.773832, 52.970321, 56.446248])

# Сдвигаем λ в положительную область для log
λ_shifted = eigenvals + 1.5  # все λ > 0

# Модель с поправкой кондуктора (логарифмическая)
def model_with_log(λ, a, b, c):
    # γ = a * λ / log(λ) + c  (как закон Вейля)
    return a * λ / np.log(λ + 1) + b * λ + c

# Обучаем на ПЕРВЫХ 6 точках
train_idx = range(6)
test_idx = range(6, 12)

try:
    popt, _ = curve_fit(model_with_log, λ_shifted[train_idx], zeros[train_idx], 
                        p0=[50, 1, 10], maxfev=5000)
    
    γ_pred_train = model_with_log(λ_shifted[train_idx], *popt)
    γ_pred_test = model_with_log(λ_shifted[test_idx], *popt)
    
    print("ПАРАМЕТРЫ (ОПРЕДЕЛЕНЫ ПО 6 ТОЧКАМ):")
    print(f"  a = {popt[0]:.3f}")
    print(f"  b = {popt[1]:.3f}")
    print(f"  c = {popt[2]:.3f}")
    
    print("\n✅ ЧЕСТНАЯ ПРОВЕРКА НА НОВЫХ (6 последних):")
    errors = []
    for i in test_idx:
        err = abs(zeros[i] - γ_pred_test[i - 6])
        errors.append(err)
        print(f"  γ_{i+1}: реальный {zeros[i]:.3f}, предск. {γ_pred_test[i-6]:.3f}, ошибка {err:.3f}")
    
    print(f"\nСРЕДНЯЯ ОШИБКА на новых: {np.mean(errors):.3f}")
    corr = np.corrcoef(zeros[test_idx], γ_pred_test)[0, 1]
    print(f"КОРРЕЛЯЦИЯ на новых: {corr:.4f}")
    
    if corr > 0.95:
        print("\n🔥🔥🔥 ПОПРАВКА КОНДУКТОРА СРАБОТАЛА! Корреляция > 0.95!")
    else:
        print(f"\n📊 Корреляция {corr:.4f} — нужно уточнять формулу.")
        
except Exception as e:
    print(f"Ошибка: {e}")
    print("Попробуем другую модель...")
    
    # Альтернативная модель: γ = a * λ * log(λ) + b
    def model_with_log2(λ, a, b, c):
        return a * λ * np.log(λ + 1) + b * λ + c
    
    popt, _ = curve_fit(model_with_log2, λ_shifted[train_idx], zeros[train_idx], 
                        p0=[10, 10, 10], maxfev=5000)
    
    γ_pred_test = model_with_log2(λ_shifted[test_idx], *popt)
    
    errors = [abs(zeros[i] - γ_pred_test[i-6]) for i in test_idx]
    corr = np.corrcoef(zeros[test_idx], γ_pred_test)[0, 1]
    
    print(f"Модель 2: γ = a * λ * log(λ) + b * λ + c")
    print(f"Средняя ошибка: {np.mean(errors):.3f}")
    print(f"Корреляция: {corr:.4f}")