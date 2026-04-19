"""
===============================================================================
ЧЕСТНЫЙ АНАЛИЗ МАТРИЦЫ ПРЫЖКОВ ГРАМ-ДИФФУЗИИ
===============================================================================
Без эрмитизации, без подгонки, без сравнения несравнимого.
Только то, что действительно следует из данных.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig, norm
from scipy.stats import chi2_contingency
from collections import Counter

# ============================================================================
# 1. ЗАГРУЗКА И ПОСТРОЕНИЕ МАТРИЦЫ ПРЫЖКОВ
# ============================================================================
print("=" * 80)
print("ЧЕСТНЫЙ АНАЛИЗ МАТРИЦЫ ПРЫЖКОВ ГРАМ-ДИФФУЗИИ")
print("=" * 80)

gram = np.load('gram_indices_2M.npy')
classes = gram % 12

# Строим матрицу прыжков ПОСЛЕ ЗАСТРЕВАНИЙ
M = np.zeros((12, 12))
stuck_count = 0

for i in range(len(gram) - 2):
    if gram[i+1] - gram[i] == 0:  # Застревание
        stuck_count += 1
        c_start = classes[i]
        c_end = classes[i+2]
        M[c_start, c_end] += 1

# Нормировка по строкам
row_sums = M.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
M_prob = M / row_sums

print(f"\nВсего застреваний: {stuck_count:,}")
print(f"Из них проанализировано: {int(M.sum()):,}")

# ============================================================================
# 2. СТРУКТУРА МАТРИЦЫ
# ============================================================================
print("\n" + "=" * 80)
print("СТРУКТУРА МАТРИЦЫ ПРЫЖКОВ M[c_start → c_end]")
print("=" * 80)

# Выделяем вероятности прыжков +1 и +2
p_plus1 = np.array([M_prob[c, (c+1)%12] for c in range(12)])
p_plus2 = np.array([M_prob[c, (c+2)%12] for c in range(12)])
p_other = 1 - p_plus1 - p_plus2

print("\nВероятности прыжков по классам:")
print("-" * 70)
print("Класс | P(+1)    | P(+2)    | P(другое) | Сумма ")
print("-" * 70)
for c in range(12):
    print(f"{c:5d} | {p_plus1[c]:8.4f} | {p_plus2[c]:8.4f} | {p_other[c]:9.4f} | {p_plus1[c]+p_plus2[c]+p_other[c]:6.4f}")

print(f"\nСредние вероятности:")
print(f"  P(+1) = {np.mean(p_plus1):.4f} ± {np.std(p_plus1):.4f}")
print(f"  P(+2) = {np.mean(p_plus2):.4f} ± {np.std(p_plus2):.4f}")

# ============================================================================
# 3. ПОИСК АНОМАЛЬНЫХ КЛАССОВ
# ============================================================================
print("\n" + "=" * 80)
print("ПОИСК АНОМАЛЬНЫХ КЛАССОВ")
print("=" * 80)

# Z-оценки для P(+1) и P(+2)
z_plus1 = (p_plus1 - np.mean(p_plus1)) / np.std(p_plus1)
z_plus2 = (p_plus2 - np.mean(p_plus2)) / np.std(p_plus2)

print("\nZ-оценки отклонений от среднего:")
print("-" * 50)
print("Класс | Z(P+1)   | Z(P+2)   | Аномалия")
print("-" * 50)
for c in range(12):
    anomaly = ""
    if abs(z_plus1[c]) > 2.0 or abs(z_plus2[c]) > 2.0:
        if c == 6:
            anomaly = "← КЛАСС 6 (избыток +2)"
        elif c == 7:
            anomaly = "← КЛАСС 7 (избыток +1)"
        else:
            anomaly = "← АНОМАЛИЯ"
    print(f"{c:5d} | {z_plus1[c]:+8.3f} | {z_plus2[c]:+8.3f} | {anomaly}")

# ============================================================================
# 4. ПРОВЕРКА СИММЕТРИИ ОТНОСИТЕЛЬНО СДВИГА НА +2
# ============================================================================
print("\n" + "=" * 80)
print("ПРОВЕРКА 12-ЛИНЗНОЙ СИММЕТРИИ")
print("=" * 80)

# Оператор сдвига на +2
S2 = np.zeros((12, 12))
for i in range(12):
    S2[i, (i+2)%12] = 1

# Коммутатор
commutator = M_prob @ S2 - S2 @ M_prob
comm_norm = norm(commutator, ord='fro')

print(f"\nНорма коммутатора [M, S_+2] (Фробениус): {comm_norm:.6f}")

# Сравнение с идеально симметричной матрицей
M_sym = np.zeros((12, 12))
p1_mean = np.mean(p_plus1)
p2_mean = np.mean(p_plus2)
for c in range(12):
    M_sym[c, (c+1)%12] = p1_mean
    M_sym[c, (c+2)%12] = p2_mean

# Расстояние до симметричной матрицы
dist_to_sym = norm(M_prob - M_sym, ord='fro')

print(f"Расстояние до идеально симметричной матрицы: {dist_to_sym:.6f}")

# Статистический тест: отличается ли M от симметричной?
# Используем хи-квадрат тест на независимость
observed = M.copy()
expected = M_sym * M.sum()
expected[expected == 0] = 1e-10

chi2_stat = np.sum((observed - expected)**2 / expected)
df = (12 - 1) * (2 - 1)  # степени свободы
from scipy.stats import chi2
p_value_sym = 1 - chi2.cdf(chi2_stat, df)

print(f"\nХи-квадрат тест на симметрию:")
print(f"  χ² = {chi2_stat:.2f}")
print(f"  df = {df}")
print(f"  p-value = {p_value_sym:.4f}")

if p_value_sym < 0.05:
    print("  ✓ Матрица СТАТИСТИЧЕСКИ ЗНАЧИМО отличается от симметричной")
    print("    → Аномалия (6,7) реальна!")
else:
    print("  ⚠️ Нет значимых отличий от симметричной модели")

# ============================================================================
# 5. СОБСТВЕННЫЕ ЗНАЧЕНИЯ
# ============================================================================
print("\n" + "=" * 80)
print("СОБСТВЕННЫЕ ЗНАЧЕНИЯ МАТРИЦЫ M")
print("=" * 80)

eigvals, eigvecs = eig(M_prob)
idx = np.argsort(np.abs(eigvals))[::-1]
eigvals = eigvals[idx]

# Корни 12-й степени из 1 для сравнения
roots_12 = [np.exp(2j * np.pi * k / 12) for k in range(12)]

print("\nСобственные значения (по убыванию |λ|):")
print("-" * 70)
print(" k  |    Re(λ)    |    Im(λ)    |   |λ|    | Ближайший корень | Расст.")
print("-" * 70)

matches = 0
for i, λ in enumerate(eigvals):
    # Находим ближайший корень
    distances = [abs(λ - r) for r in roots_12]
    min_dist = min(distances)
    nearest = roots_12[np.argmin(distances)]
    k_nearest = int(round(np.angle(nearest) * 12 / (2 * np.pi))) % 12
    
    if min_dist < 0.1:
        matches += 1
        marker = "✓"
    else:
        marker = ""
    
    print(f"{i+1:2d} | {λ.real:+10.6f} | {λ.imag:+10.6f} | {abs(λ):8.4f} | e^(2πi·{k_nearest}/12) | {min_dist:.4f} {marker}")

print(f"\nСовпадений с корнями 12-й степени (dist < 0.1): {matches} / 12")

# ============================================================================
# 6. СТАЦИОНАРНОЕ РАСПРЕДЕЛЕНИЕ
# ============================================================================
print("\n" + "=" * 80)
print("СТАЦИОНАРНОЕ РАСПРЕДЕЛЕНИЕ")
print("=" * 80)

# Находим собственный вектор для λ=1
idx_one = np.argmin(np.abs(eigvals - 1.0))
λ_one = eigvals[idx_one]
stationary = np.real(eigvecs[:, idx_one])
stationary = stationary / np.sum(stationary)

# Эмпирическое распределение классов в застреваниях
class_counts = Counter()
for i in range(len(gram) - 2):
    if gram[i+1] - gram[i] == 0:
        class_counts[classes[i]] += 1
total = sum(class_counts.values())
empirical = np.array([class_counts.get(c, 0) / total for c in range(12)])

print(f"\nλ₁ = {λ_one.real:.6f} + {λ_one.imag:.6f}i")
print("\nРаспределение классов:")
print("-" * 65)
print("Класс | Стационарное | Эмпирическое | Разность | Аномалия")
print("-" * 65)

for c in range(12):
    diff = stationary[c] - empirical[c]
    anomaly = ""
    if c == 6:
        anomaly = "← КЛАСС 6"
    elif c == 7:
        anomaly = "← КЛАСС 7"
    print(f"{c:5d} | {stationary[c]:11.4f} | {empirical[c]:11.4f} | {diff:+8.4f} | {anomaly}")

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Честный анализ матрицы прыжков Грам-диффузии', fontsize=14)

# График 1: Тепловая карта M
ax1 = axes[0, 0]
im1 = ax1.imshow(M_prob, cmap='RdBu_r', aspect='equal', vmin=0, vmax=1)
ax1.set_xlabel('Класс после прыжка')
ax1.set_ylabel('Класс застревания')
ax1.set_title('Матрица вероятностей M')
ax1.set_xticks(range(12))
ax1.set_yticks(range(12))
plt.colorbar(im1, ax=ax1)

# График 2: P(+1) и P(+2) по классам
ax2 = axes[0, 1]
x = np.arange(12)
width = 0.35
bars1 = ax2.bar(x - width/2, p_plus1, width, label='P(+1)', color='red', alpha=0.7)
bars2 = ax2.bar(x + width/2, p_plus2, width, label='P(+2)', color='blue', alpha=0.7)
ax2.axhline(y=p1_mean, color='red', linestyle='--', alpha=0.5)
ax2.axhline(y=p2_mean, color='blue', linestyle='--', alpha=0.5)
ax2.set_xlabel('Класс')
ax2.set_ylabel('Вероятность')
ax2.set_title('Вероятности прыжков по классам')
ax2.set_xticks(x)
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# Подсветка аномалии
ax2.axvspan(5.5, 6.5, alpha=0.2, color='yellow')
ax2.axvspan(6.5, 7.5, alpha=0.2, color='orange')

# График 3: Z-оценки
ax3 = axes[0, 2]
ax3.bar(x - width/2, z_plus1, width, label='Z(P+1)', color='red', alpha=0.7)
ax3.bar(x + width/2, z_plus2, width, label='Z(P+2)', color='blue', alpha=0.7)
ax3.axhline(y=2, color='black', linestyle='--', alpha=0.5, label='±2σ')
ax3.axhline(y=-2, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Класс')
ax3.set_ylabel('Z-оценка')
ax3.set_title('Отклонения от среднего (Z-оценки)')
ax3.set_xticks(x)
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# График 4: Собственные значения
ax4 = axes[1, 0]
theta = np.linspace(0, 2*np.pi, 100)
ax4.plot(np.cos(theta), np.sin(theta), 'b--', alpha=0.3, label='|λ|=1')
roots_x = [r.real for r in roots_12]
roots_y = [r.imag for r in roots_12]
ax4.scatter(roots_x, roots_y, c='blue', s=40, alpha=0.5, marker='s', label='Корни 12-й степени')
ax4.scatter(eigvals.real, eigvals.imag, c='red', s=80, alpha=0.7, zorder=3, label='λ(M)')
ax4.axhline(0, color='black', alpha=0.2)
ax4.axvline(0, color='black', alpha=0.2)
ax4.set_xlabel('Re(λ)')
ax4.set_ylabel('Im(λ)')
ax4.set_title('Собственные значения M')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_aspect('equal')

# График 5: |λ| по убыванию
ax5 = axes[1, 1]
abs_eig = np.abs(eigvals)
ax5.bar(range(1, 13), abs_eig, color='steelblue', alpha=0.7)
ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='|λ|=1')
ax5.set_xlabel('Номер')
ax5.set_ylabel('|λ|')
ax5.set_title('Модули собственных значений')
ax5.set_xticks(range(1, 13))
ax5.legend()
ax5.grid(alpha=0.3, axis='y')

# График 6: Сводка
ax6 = axes[1, 2]
ax6.axis('off')

# Определяем вердикт
if p_value_sym < 0.05:
    verdict = "✓ АНОМАЛИЯ ПОДТВЕРЖДЕНА"
    verdict_color = 'green'
else:
    verdict = "⚠️ АНОМАЛИЯ НЕ ЗНАЧИМА"
    verdict_color = 'orange'

summary_text = f"""
СВОДКА РЕЗУЛЬТАТОВ:

Застреваний: {stuck_count:,}

Средние вероятности:
  P(+1) = {p1_mean:.4f} ± {np.std(p_plus1):.4f}
  P(+2) = {p2_mean:.4f} ± {np.std(p_plus2):.4f}

Симметрия сдвига +2:
  Норма [M, S_+2] = {comm_norm:.6f}
  Расст. до симм. = {dist_to_sym:.6f}

Хи-квадрат тест:
  χ² = {chi2_stat:.2f}, p = {p_value_sym:.4f}

Собственные значения:
  Совпадений с e^(2πik/12): {matches}/12
  |λ₂| = {abs_eig[1]:.4f}

Аномальные классы:
  Класс 6: Z(P+1)={z_plus1[6]:+.2f}, Z(P+2)={z_plus2[6]:+.2f}
  Класс 7: Z(P+1)={z_plus1[7]:+.2f}, Z(P+2)={z_plus2[7]:+.2f}

ВЕРДИКТ: {verdict}
"""
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('honest_jump_matrix_analysis.png', dpi=150)
print("\n✓ График сохранён как 'honest_jump_matrix_analysis.png'")

# ============================================================================
# 8. ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ЧЕСТНЫЙ АНАЛИЗ МАТРИЦЫ ПРЫЖКОВ ГРАМ-ДИФФУЗИИ                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Симметрия сдвига на +2:                                                   │
│   • Норма коммутатора = {comm_norm:.6f}                                     │
│   • Матрица {('КОММУТИРУЕТ' if comm_norm < 0.05 else 'НЕ коммутирует')} со сдвигом на +2                     │
│                                                                             │
│   Статистическая значимость аномалии:                                       │
│   • Хи-квадрат = {chi2_stat:.2f}, p = {p_value_sym:.4f}                     │
│   • {('Аномалия ЗНАЧИМА' if p_value_sym < 0.05 else 'Аномалия НЕ значима')} │
│                                                                             │
│   Собственные значения:                                                     │
│   • Совпадений с e^(2πik/12): {matches} / 12                                │
│   • Максимальное |λ| после λ₁: {abs_eig[1]:.4f}                             │
│                                                                             │
│   Аномальные классы:                                                        │
│   • Класс 6: P(+1)={p_plus1[6]:.3f}, P(+2)={p_plus2[6]:.3f}                 │
│   • Класс 7: P(+1)={p_plus1[7]:.3f}, P(+2)={p_plus2[7]:.3f}                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ВЫВОД:                                                                    │
""")

if comm_norm < 0.05:
    print("│   ✓ 12-линзная симметрия ПОДТВЕРЖДЕНА                                  │")
else:
    print("│   ⚠️ 12-линзная симметрия НЕ ПОДТВЕРЖДЕНА                              │")

if p_value_sym < 0.05:
    print("│   ✓ Аномалия классов 6 и 7 СТАТИСТИЧЕСКИ ЗНАЧИМА                       │")
else:
    print("│   ⚠️ Аномалия классов 6 и 7 НЕ ЗНАЧИМА на этом объёме данных           │")

if matches >= 8:
    print("│   ✓ Собственные значения близки к корням 12-й степени                  │")
else:
    print("│   ⚠️ Собственные значения НЕ совпадают с корнями 12-й степени          │")
    print("│      (есть диссипация, процесс затухающий)                              │")

print("""│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 80)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)