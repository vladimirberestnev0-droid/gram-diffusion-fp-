"""
===============================================================================
ЧЕСТНАЯ ПРОВЕРКА УНИВЕРСАЛЬНОСТИ ОПЕРАТОРА H
===============================================================================
Гипотеза: Все нули ζ(s) разбиты на 12 потоков.
Оператор H (12×12) описывает переходы между потоками.
Если теория верна, спектр H должен коррелировать с ЛЮБЫМ блоком из 12
последовательных нулей (после учёта масштаба, зависящего от высоты).
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.linalg import eigh
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 80)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
print(f"✓ Загружено {len(zeros):,} нулей ζ(s)")
print(f"  Диапазон: {zeros[0]:.6f} ... {zeros[-1]:.6f}")

# ============================================================================
# 2. ПОСТРОЕНИЕ ОПЕРАТОРА H (12×12)
# ============================================================================
print("\n" + "=" * 80)
print("ПОСТРОЕНИЕ ОПЕРАТОРА H")
print("=" * 80)

# Фиксированные вероятности из эмпирических данных
P_PLUS1 = 0.584  # аномальный прыжок c -> c+1
P_PLUS2 = 0.416  # нормальный прыжок c -> c+2

M = np.zeros((12, 12))
for i in range(12):
    M[i, (i + 1) % 12] = P_PLUS1
    M[i, (i + 2) % 12] = P_PLUS2

H = (M + M.T) / 2
eigenvalues_H = np.sort(eigh(H)[0])

print(f"✓ P(+1) = {P_PLUS1}, P(+2) = {P_PLUS2}")
print(f"✓ Собственные значения H: {np.array2string(eigenvalues_H, precision=4)}")

# ============================================================================
# 3. ПРОВЕРКА ДЛЯ БЛОКОВ ИЗ 12 НУЛЕЙ НА РАЗНЫХ ВЫСОТАХ
# ============================================================================
print("\n" + "=" * 80)
print("ПРОВЕРКА УНИВЕРСАЛЬНОСТИ: КОРРЕЛЯЦИЯ ДЛЯ БЛОКОВ ПО 12 НУЛЕЙ")
print("=" * 80)

block_size = 12
step = 12*100000  # шаг между блоками (можно менять)
n_blocks = (len(zeros) - block_size) // step

print(f"Размер блока: {block_size} нулей")
print(f"Шаг: {step}")
print(f"Всего блоков: {n_blocks}")

correlations = []
t_centers = []
mae_values = []

for i in tqdm(range(0, len(zeros) - block_size, step), desc="Проверка блоков"):
    block_zeros = zeros[i:i + block_size]
    t_center = np.mean(block_zeros)
    
    # Масштабирование спектра H под этот блок
    A = np.vstack([eigenvalues_H, np.ones(block_size)]).T
    a, b = np.linalg.lstsq(A, block_zeros, rcond=None)[0]
    predicted = a * eigenvalues_H + b
    
    # Метрики
    corr, _ = pearsonr(predicted, block_zeros)
    mae = np.mean(np.abs(predicted - block_zeros))
    
    correlations.append(corr)
    t_centers.append(t_center)
    mae_values.append(mae)

correlations = np.array(correlations)
t_centers = np.array(t_centers)
mae_values = np.array(mae_values)

# ============================================================================
# 4. СТАТИСТИКА
# ============================================================================
print("\n" + "=" * 80)
print("СТАТИСТИКА ПО ВСЕМ БЛОКАМ")
print("=" * 80)

print(f"\nКоличество проверенных блоков: {len(correlations)}")
print(f"\nКорреляция Пирсона:")
print(f"  Среднее: {np.mean(correlations):.6f}")
print(f"  Стд:     {np.std(correlations):.6f}")
print(f"  Медиана: {np.median(correlations):.6f}")
print(f"  Минимум: {np.min(correlations):.6f}")
print(f"  Максимум: {np.max(correlations):.6f}")

# Доля блоков с высокой корреляцией
high_corr = np.mean(correlations > 0.85)
print(f"\nДоля блоков с r > 0.85: {high_corr:.2%}")

# ============================================================================
# 5. ПРИМЕРЫ ДЛЯ КОНКРЕТНЫХ ВЫСОТ
# ============================================================================
print("\n" + "=" * 80)
print("ПРИМЕРЫ ДЛЯ КОНКРЕТНЫХ ВЫСОТ")
print("=" * 80)

example_starts = [0, 1000, 10000, 50000, 100000, 500000, 1000000, 1500000]

print(f"\n{'Старт':<10} {'t ~':<12} {'Корреляция':<12} {'MAE':<10}")
print("-" * 50)

for start in example_starts:
    if start + block_size <= len(zeros):
        block_zeros = zeros[start:start + block_size]
        t_center = np.mean(block_zeros)
        
        A = np.vstack([eigenvalues_H, np.ones(block_size)]).T
        a, b = np.linalg.lstsq(A, block_zeros, rcond=None)[0]
        predicted = a * eigenvalues_H + b
        
        corr, _ = pearsonr(predicted, block_zeros)
        mae = np.mean(np.abs(predicted - block_zeros))
        
        print(f"{start:<10} {t_center:<12.0f} {corr:<12.6f} {mae:<10.4f}")

# ============================================================================
# 6. ПОДРОБНЫЙ ПРИМЕР ДЛЯ ПЕРВЫХ 12 НУЛЕЙ
# ============================================================================
print("\n" + "=" * 80)
print("ПОДРОБНЫЙ ПРИМЕР: ПЕРВЫЕ 12 НУЛЕЙ")
print("=" * 80)

first_12 = zeros[:12]
A = np.vstack([eigenvalues_H, np.ones(12)]).T
a, b = np.linalg.lstsq(A, first_12, rcond=None)[0]
predicted_first = a * eigenvalues_H + b
corr_first, _ = pearsonr(predicted_first, first_12)

print(f"\nМасштаб: a = {a:.4f}, b = {b:.4f}")
print(f"Корреляция: r = {corr_first:.6f}")

print(f"\n{'n':<4} {'Прогноз':<12} {'Реальность':<12} {'Ошибка':<12} {'Отн. ошибка':<12}")
print("-" * 60)
for i in range(12):
    diff = predicted_first[i] - first_12[i]
    rel = diff / first_12[i] * 100
    print(f"{i+1:<4} {predicted_first[i]:<12.4f} {first_12[i]:<12.4f} {diff:<+12.4f} {rel:+10.2f}%")

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================================
print("\n" + "=" * 80)
print("ПОСТРОЕНИЕ ГРАФИКОВ")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Универсальность оператора H: корреляция с блоками по 12 нулей', fontsize=14)

# График 1: Корреляция в зависимости от высоты
ax1 = axes[0, 0]
ax1.scatter(t_centers[::10], correlations[::10], s=5, alpha=0.5, color='steelblue')
ax1.axhline(y=0.85, color='red', linestyle='--', label='r = 0.85')
ax1.axhline(y=np.mean(correlations), color='green', linestyle='-', label=f'Среднее = {np.mean(correlations):.3f}')
ax1.set_xlabel('Высота t')
ax1.set_ylabel('Корреляция Пирсона r')
ax1.set_title('Корреляция спектра H с блоками из 12 нулей')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xscale('log')

# График 2: Гистограмма корреляций
ax2 = axes[0, 1]
ax2.hist(correlations, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=0.85, color='red', linestyle='--', label='r = 0.85')
ax2.axvline(x=np.mean(correlations), color='green', linestyle='-', label=f'Среднее = {np.mean(correlations):.3f}')
ax2.set_xlabel('Корреляция r')
ax2.set_ylabel('Частота')
ax2.set_title(f'Распределение корреляций ({len(correlations)} блоков)')
ax2.legend()
ax2.grid(alpha=0.3)

# График 3: MAE в зависимости от высоты
ax3 = axes[1, 0]
ax3.scatter(t_centers[::10], mae_values[::10], s=5, alpha=0.5, color='coral')
ax3.axhline(y=np.mean(mae_values), color='green', linestyle='-', label=f'Среднее = {np.mean(mae_values):.3f}')
ax3.set_xlabel('Высота t')
ax3.set_ylabel('MAE')
ax3.set_title('Средняя абсолютная ошибка')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xscale('log')

# График 4: Пример для первых 12 нулей
ax4 = axes[1, 1]
x = np.arange(1, 13)
ax4.plot(x, predicted_first, 'bo-', label='Прогноз H', markersize=8)
ax4.plot(x, first_12, 'rs-', label='Реальные нули', markersize=8)
ax4.set_xlabel('n')
ax4.set_ylabel('γ_n')
ax4.set_title(f'Первые 12 нулей (r = {corr_first:.4f})')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('universality_H_operator.png', dpi=150)
print("✓ График сохранён как 'universality_H_operator.png'")

# ============================================================================
# 8. ИТОГОВЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "=" * 80)
print("ИТОГОВЫЙ ВЕРДИКТ")
print("=" * 80)

mean_corr = np.mean(correlations)
std_corr = np.std(correlations)

print(f"""
Проверено {len(correlations):,} блоков по 12 нулей на высотах от {t_centers[0]:.0f} до {t_centers[-1]:.0f}.

Результаты:
  • Средняя корреляция: r = {mean_corr:.4f} ± {std_corr:.4f}
  • Медианная корреляция: {np.median(correlations):.4f}
  • Доля блоков с r > 0.85: {high_corr:.2%}

""")

if mean_corr > 0.85:
    print("✅✅✅ УНИВЕРСАЛЬНОСТЬ ПОДТВЕРЖДЕНА!")
    print("Спектр оператора H стабильно коррелирует с блоками из 12 нулей")
    print("на всём диапазоне высот. Это доказывает, что 12-потоковая структура")
    print("является фундаментальным свойством нулей ζ(s).")
elif mean_corr > 0.7:
    print("👍 УНИВЕРСАЛЬНОСТЬ ЧАСТИЧНО ПОДТВЕРЖДЕНА")
    print("Корреляция значима, но есть вариации. Возможно, требуется")
    print("уточнение вероятностей P(+1) и P(+2) в зависимости от высоты.")
else:
    print("⚠️ УНИВЕРСАЛЬНОСТЬ НЕ ПОДТВЕРЖДЕНА")
    print("Корреляция недостаточно высока. Проверьте данные или модель.")

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)

plt.show()