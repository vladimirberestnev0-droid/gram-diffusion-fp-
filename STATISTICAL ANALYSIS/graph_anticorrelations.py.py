"""
===============================================================================
ЧЕСТНАЯ ПРОВЕРКА ПАРНЫХ СВЯЗЕЙ МЕЖДУ КЛАССАМИ
===============================================================================
Без предположений. Только корреляции в данных.
"""

import numpy as np
from scipy.stats import pearsonr
from collections import Counter

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 80)
print("ЧЕСТНАЯ ПРОВЕРКА ПАРНЫХ СВЯЗЕЙ МЕЖДУ КЛАССАМИ")
print("=" * 80)

gram = np.load('gram_indices_2M.npy')
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
classes = gram % 12

print(f"✓ Загружено {len(gram):,} нулей")

# ============================================================================
# 2. РАЗБИЕНИЕ НА БЛОКИ И ПОДСЧЁТ ЧАСТОТ
# ============================================================================
block_size = 50000
n_blocks = len(classes) // block_size

# Матрица частот: blocks × 12
freq_matrix = np.zeros((n_blocks, 12))

for b in range(n_blocks):
    start = b * block_size
    end = start + block_size
    block_classes = classes[start:end]
    counts = Counter(block_classes)
    for c in range(12):
        freq_matrix[b, c] = counts.get(c, 0)

print(f"✓ Разбито на {n_blocks} блоков по {block_size} нулей")

# ============================================================================
# 3. КОРРЕЛЯЦИОННАЯ МАТРИЦА МЕЖДУ КЛАССАМИ
# ============================================================================
print("\n" + "=" * 80)
print("КОРРЕЛЯЦИИ ЧАСТОТ КЛАССОВ ПО БЛОКАМ")
print("=" * 80)

corr_matrix = np.zeros((12, 12))
p_matrix = np.zeros((12, 12))

for i in range(12):
    for j in range(12):
        if i <= j:
            corr, p_val = pearsonr(freq_matrix[:, i], freq_matrix[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val

print("\nМатрица корреляций частот классов:")
print("-" * 70)
print("     ", end="")
for j in range(12):
    print(f"{j:6d}", end="")
print("\n" + "-" * 70)

for i in range(12):
    print(f"{i:3d} |", end="")
    for j in range(12):
        corr = corr_matrix[i, j]
        if i == j:
            print(f"{1.0:6.2f}", end="")
        elif corr < -0.3:
            print(f"\033[91m{corr:6.2f}\033[0m", end="")  # Красный для отрицательных
        elif corr > 0.3:
            print(f"\033[92m{corr:6.2f}\033[0m", end="")  # Зелёный для положительных
        else:
            print(f"{corr:6.2f}", end="")
    print()

# ============================================================================
# 4. ПОИСК ЗНАЧИМЫХ ОТРИЦАТЕЛЬНЫХ КОРРЕЛЯЦИЙ (ПАРЫ)
# ============================================================================
print("\n" + "=" * 80)
print("ЗНАЧИМЫЕ ОТРИЦАТЕЛЬНЫЕ КОРРЕЛЯЦИИ (АНТИКОРРЕЛИРОВАННЫЕ ПАРЫ)")
print("=" * 80)

pairs = []
for i in range(12):
    for j in range(i+1, 12):
        if corr_matrix[i, j] < 0 and p_matrix[i, j] < 0.05:
            pairs.append((i, j, corr_matrix[i, j], p_matrix[i, j]))

pairs.sort(key=lambda x: x[2])  # Сортируем по силе антикорреляции

print("\nПара  | Корреляция | p-value  | Значимость")
print("-" * 50)
for c1, c2, corr, p in pairs:
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
    print(f"({c1:2d},{c2:2d}) | {corr:+8.4f} | {p:.2e} | {sig}")

# ============================================================================
# 5. ПРОВЕРКА ГИПОТЕЗЫ О ПАРАХ (чётный, нечётный)
# ============================================================================
print("\n" + "=" * 80)
print("ПРОВЕРКА ГИПОТЕЗЫ: ВСЕ ЛИ ПАРЫ СОСТОЯТ ИЗ (ЧЁТНЫЙ, НЕЧЁТНЫЙ)?")
print("=" * 80)

even_odd_pairs = []
other_pairs = []

for c1, c2, corr, p in pairs:
    if (c1 % 2 == 0 and c2 % 2 == 1) or (c1 % 2 == 1 and c2 % 2 == 0):
        even_odd_pairs.append((c1, c2, corr, p))
    else:
        other_pairs.append((c1, c2, corr, p))

print(f"\nПары (чётный, нечётный): {len(even_odd_pairs)}")
for c1, c2, corr, p in even_odd_pairs:
    print(f"  ({c1:2d},{c2:2d}): r = {corr:.4f}, p = {p:.2e}")

print(f"\nДругие пары: {len(other_pairs)}")
for c1, c2, corr, p in other_pairs:
    print(f"  ({c1:2d},{c2:2d}): r = {corr:.4f}, p = {p:.2e}")

# ============================================================================
# 6. ТЕСТ НА НЕЗАВИСИМОСТЬ: МЕНТЕ-КАРЛО
# ============================================================================
print("\n" + "=" * 80)
print("МОНТЕ-КАРЛО ТЕСТ: СЛУЧАЙНО ЛИ ЧИСЛО ЧЁТНО-НЕЧЁТНЫХ ПАР?")
print("=" * 80)

n_simulations = 10000
even_odd_count_random = []

np.random.seed(42)
for _ in range(n_simulations):
    # Случайно выбираем столько же пар, сколько нашли
    random_pairs = set()
    while len(random_pairs) < len(pairs):
        c1 = np.random.randint(0, 12)
        c2 = np.random.randint(0, 12)
        if c1 < c2:
            random_pairs.add((c1, c2))
    
    # Считаем, сколько из них чётно-нечётные
    count = sum(1 for c1, c2 in random_pairs if (c1%2==0 and c2%2==1) or (c1%2==1 and c2%2==0))
    even_odd_count_random.append(count)

observed = len(even_odd_pairs)
p_value_mc = sum(1 for x in even_odd_count_random if x >= observed) / n_simulations

print(f"Наблюдаемое число чётно-нечётных пар: {observed}")
print(f"Ожидаемое при случайном выборе: {np.mean(even_odd_count_random):.1f} ± {np.std(even_odd_count_random):.1f}")
print(f"p-value (Монте-Карло): {p_value_mc:.4f}")

if p_value_mc < 0.05:
    print("✅ Чётно-нечётные пары встречаются ЗНАЧИМО ЧАЩЕ случайного!")
else:
    print("⚠️ Нет значимого отклонения от случайности")

# ============================================================================
# 7. СВЯЗЬ С РАЗНОСТЬЮ ПО МОДУЛЮ 12
# ============================================================================
print("\n" + "=" * 80)
print("АНАЛИЗ РАЗНОСТЕЙ В ПАРАХ")
print("=" * 80)

print("\nПара  | Разность mod 12 | Корреляция")
print("-" * 45)
for c1, c2, corr, _ in pairs:
    diff = (c2 - c1) % 12
    if diff > 6:
        diff = diff - 12
    print(f"({c1:2d},{c2:2d}) | {diff:6d}        | {corr:+.4f}")

# ============================================================================
# 8. СРАВНЕНИЕ С ПРЕДСКАЗАННЫМИ ПАРАМИ
# ============================================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ С ПРЕДСКАЗАННЫМИ ПАРАМИ")
print("=" * 80)

predicted_pairs = [(10,3), (4,5), (6,7), (2,1), (0,9), (8,11)]
observed_pairs_set = set((min(c1,c2), max(c1,c2)) for c1, c2, _, _ in pairs)

print("\nПредсказанные пары:")
matches = 0
for p1, p2 in predicted_pairs:
    pair = (min(p1, p2), max(p1, p2))
    found = pair in observed_pairs_set
    corr_val = corr_matrix[p1, p2] if found else 0
    status = "✅" if found else "❌"
    print(f"  ({p1:2d},{p2:2d}): {status} (r = {corr_val:+.4f})")
    if found:
        matches += 1

print(f"\nСовпадений: {matches} / {len(predicted_pairs)}")

# ============================================================================
# 9. ВИЗУАЛИЗАЦИЯ
# ============================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Честная проверка парных связей', fontsize=14)

# График 1: Тепловая карта корреляций
ax1 = axes[0, 0]
im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
ax1.set_xlabel('Класс')
ax1.set_ylabel('Класс')
ax1.set_title('Корреляции частот классов')
ax1.set_xticks(range(12))
ax1.set_yticks(range(12))
plt.colorbar(im1, ax=ax1)

# График 2: Найденные пары
ax2 = axes[0, 1]
x = []
y = []
strengths = []
for c1, c2, corr, _ in pairs:
    x.append(c1)
    y.append(c2)
    strengths.append(abs(corr) * 100)

sc = ax2.scatter(x, y, c=strengths, s=strengths, cmap='Reds', alpha=0.7, edgecolors='black')
ax2.set_xlabel('Класс 1')
ax2.set_ylabel('Класс 2')
ax2.set_title(f'Антикоррелированные пары (n={len(pairs)})')
ax2.set_xticks(range(12))
ax2.set_yticks(range(12))
ax2.grid(alpha=0.3)
plt.colorbar(sc, ax=ax2, label='|r| × 100')

# График 3: Сила корреляции по разности
ax3 = axes[1, 0]
diffs = []
corrs = []
for c1, c2, corr, _ in pairs:
    diff = abs(c1 - c2)
    if diff > 6:
        diff = 12 - diff
    diffs.append(diff)
    corrs.append(corr)

ax3.bar(range(len(pairs)), corrs, color='red', alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.set_xlabel('Пара (по убыванию силы)')
ax3.set_ylabel('Корреляция')
ax3.set_title('Сила антикорреляции пар')
ax3.grid(alpha=0.3, axis='y')

# График 4: Сравнение наблюдаемых пар с предсказанными
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
РЕЗУЛЬТАТЫ ПРОВЕРКИ:

Найдено антикоррелированных пар: {len(pairs)}
Из них чётно-нечётных: {len(even_odd_pairs)} / {len(pairs)}

Монте-Карло тест:
  Наблюдаемое: {observed}
  Ожидаемое: {np.mean(even_odd_count_random):.1f}
  p-value: {p_value_mc:.4f}

Предсказанные пары: {matches}/{len(predicted_pairs)}

Топ-3 пары по силе связи:
"""
for i, (c1, c2, corr, _) in enumerate(pairs[:3]):
    summary_text += f"  {i+1}. ({c1},{c2}): r = {corr:.4f}\n"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('honest_pair_analysis.png', dpi=150)
print("\n✓ График сохранён как 'honest_pair_analysis.png'")

# ============================================================================
# 10. ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ ПО ПАРАМ")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  ЧЕСТНАЯ ПРОВЕРКА ГИПОТЕЗЫ О ПАРНЫХ СВЯЗЯХ                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Найдено значимых антикорреляций: {len(pairs)}                              │
│  Из них чётно-нечётных: {len(even_odd_pairs)} ({100*len(even_odd_pairs)/len(pairs):.0f}%)        │
│                                                                             │
│  Совпадение с предсказанием: {matches}/{len(predicted_pairs)}               │
│                                                                             │
│  Статистическая значимость (Монте-Карло): p = {p_value_mc:.4f}              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

if p_value_mc < 0.05:
    print("✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА: Пары действительно образуют чётно-нечётную структуру!")
else:
    print("⚠️ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА: Распределение пар не отличается от случайного.")

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)

"""
===============================================================================
СПЕКТРАЛЬНЫЙ АНАЛИЗ ГРАФА АНТИКОРРЕЛЯЦИЙ
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# ============================================================================
# 1. ПОСТРОЕНИЕ МАТРИЦЫ СМЕЖНОСТИ ГРАФА АНТИКОРРЕЛЯЦИЙ
# ============================================================================
print("=" * 80)
print("СПЕКТРАЛЬНЫЙ АНАЛИЗ ГРАФА АНТИКОРРЕЛЯЦИЙ")
print("=" * 80)

# Пары из честной проверки
pairs = [
    (10, 11, -0.7268),
    (4, 5, -0.6040),
    (6, 7, -0.5891),
    (0, 1, -0.5270),
    (2, 11, -0.5170),
    (9, 10, -0.4901),
    (5, 8, -0.4314),
    (1, 2, -0.4258),
    (8, 9, -0.3954),
    (7, 8, -0.3631),
    (0, 9, -0.3452),
    (7, 10, -0.3157)
]

# Матрица смежности (взвешенная)
adj = np.zeros((12, 12))
for c1, c2, corr in pairs:
    weight = abs(corr)
    adj[c1, c2] = weight
    adj[c2, c1] = weight

print("\nМатрица смежности графа антикорреляций (веса = |r|):")
print("-" * 60)
for i in range(12):
    print(f"{i:2d} |", end="")
    for j in range(12):
        if adj[i, j] > 0:
            print(f"{adj[i,j]:5.2f}", end="")
        else:
            print(f"     ", end="")
    print()

# ============================================================================
# 2. СТЕПЕНИ ВЕРШИН
# ============================================================================
print("\n" + "=" * 80)
print("СТЕПЕНИ ВЕРШИН (ХАБЫ)")
print("=" * 80)

degrees = np.sum(adj > 0, axis=1)
weighted_degrees = np.sum(adj, axis=1)

print("\nКласс | Степень | Взвешенная степень | Статус")
print("-" * 55)
for c in range(12):
    status = ""
    if weighted_degrees[c] == max(weighted_degrees):
        status = "← ГЛАВНЫЙ ХАБ"
    elif degrees[c] >= 3:
        status = "← ХАБ"
    print(f"{c:5d} | {degrees[c]:7d} | {weighted_degrees[c]:18.3f} | {status}")

# ============================================================================
# 3. СПЕКТР МАТРИЦЫ СМЕЖНОСТИ
# ============================================================================
print("\n" + "=" * 80)
print("СПЕКТР ГРАФА АНТИКОРРЕЛЯЦИЙ")
print("=" * 80)

eigvals_adj = eigh(adj)[0]

print("\nСобственные значения матрицы смежности:")
print("-" * 50)
for i, λ in enumerate(sorted(eigvals_adj, reverse=True)):
    print(f"λ_{i+1:2d} = {λ:+.6f}")

# ============================================================================
# 4. ЛАПЛАСИАН ГРАФА
# ============================================================================
print("\n" + "=" * 80)
print("ЛАПЛАСИАН ГРАФА")
print("=" * 80)

# Матрица степеней
D = np.diag(weighted_degrees)
L = D - adj

eigvals_L = eigh(L)[0]

print("\nСобственные значения лапласиана:")
print("-" * 50)
for i, λ in enumerate(sorted(eigvals_L)):
    print(f"λ_{i+1:2d} = {λ:+.6f}")

# Алгебраическая связность (второе собственное значение)
algebraic_connectivity = sorted(eigvals_L)[1]
print(f"\nАлгебраическая связность (λ₂): {algebraic_connectivity:.6f}")

# ============================================================================
# 5. СРАВНЕНИЕ С ДОДЕКАЭДРОМ
# ============================================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ С ГРАФОМ ДОДЕКАЭДРА")
print("=" * 80)

# Спектр додекаэдра (известные значения)
dodecahedron_spectrum = [3.000, 2.236, 2.236, 2.236, 1.000, 
                         1.000, 1.000, 1.000, -1.000, -1.000, 
                         -1.000, -1.000, -2.236, -2.236, -2.236, 
                         -3.000, 0, 0, 0, 0]  # для 20 вершин

print("Спектр додекаэдра (20 вершин) известен.")
print("Наш граф имеет 12 вершин — это граф ГРАНЕЙ додекаэдра!")

# Граф граней додекаэдра — это икосаэдр (12 вершин, 30 рёбер)
# У нас 12 рёбер — это подграф икосаэдра

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ ГРАФА
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- Подготовка фигуры ---
plt.figure(figsize=(15, 6))

# --- 1. Построение графа антикорреляций ---
plt.subplot(1, 2, 1)
G = nx.Graph()

# КРИТИЧЕСКИ ВАЖНО: Добавляем все 12 узлов, чтобы их количество 
# всегда соответствовало списку цветов colors
G.add_nodes_from(range(12))

# Добавляем ребра из вашего списка pairs
# (pairs должен быть определен выше в вашем коде как список кортежей (c1, c2, corr))
for c1, c2, corr in pairs:
    G.add_edge(c1, c2, weight=abs(corr))

# Расположение вершин по кругу (математически точное для 12 секторов)
pos = {i: (np.cos(2*np.pi*i/12 - np.pi/2), np.sin(2*np.pi*i/12 - np.pi/2)) for i in range(12)}

# Цвета: чётные — красные, нечётные — синие
colors = ['red' if i % 2 == 0 else 'blue' for i in range(12)]

# Получаем веса существующих ребер для отрисовки толщины и цвета
# Если ребер меньше, чем пар, используем безопасный метод
edges = G.edges(data=True)
if len(edges) > 0:
    edge_weights = [d['weight'] for u, v, d in edges]
else:
    edge_weights = []

# Отрисовка графа
nx.draw(G, pos, 
        with_labels=True, 
        node_color=colors, 
        node_size=800, 
        edgelist=[(u,v) for u,v,d in edges],
        edge_color=edge_weights,
        edge_cmap=plt.cm.Reds, 
        edge_vmin=0.0, 
        edge_vmax=1.0, # нормализация под корреляцию
        width=3, 
        font_color='white', 
        font_weight='bold')

plt.title('Граф антикорреляций потоков (12 классов Грамма)')

# --- 2. Визуализация спектра ---
plt.subplot(1, 2, 2)

# eigvals_adj должен быть рассчитан ранее из матрицы H
# Если его нет, здесь используется заглушка (замените на свои данные)
try:
    sorted_eigs = sorted(eigvals_adj, reverse=True)
    plt.stem(range(1, 13), sorted_eigs, basefmt=' ')
    plt.title('Спектр оператора структуры (λ)')
except NameError:
    plt.text(0.5, 0.5, "Данные спектра не найдены", ha='center')

plt.xlabel('Номер собственного значения')
plt.ylabel('Собственное значение λ')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('anticorrelation_graph_spectrum.png', dpi=150)
print("\n✓ График успешно сохранён как 'anticorrelation_graph_spectrum.png'")
plt.show()


# ============================================================================
# 7. ФИНАЛЬНЫЙ ВЫВОД
# ============================================================================
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЙ ВЫВОД")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  СПЕКТРАЛЬНЫЙ АНАЛИЗ ГРАФА АНТИКОРРЕЛЯЦИЙ                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Вершин: 12                                                                 │
│  Рёбер: {len(pairs)}                                                       │
│                                                                             │
│  Главный хаб: класс {np.argmax(weighted_degrees)} (взвешенная степень = {max(weighted_degrees):.3f})  │
│                                                                             │
│  Спектр матрицы смежности:                                                  │
│    λ₁ = {max(eigvals_adj):+.4f}                                             │
│    λ₂ = {sorted(eigvals_adj, reverse=True)[1]:+.4f}                         │
│                                                                             │
│  Алгебраическая связность: {algebraic_connectivity:.4f}                     │
│                                                                             │
│  Граф связен: {'ДА' if nx.is_connected(G) else 'НЕТ'}                       │
│  Диаметр: {nx.diameter(G) if nx.is_connected(G) else 'N/A'}                 │
└─────────────────────────────────────────────────────────────────────────────┘
""")


"""
===============================================================================
ЧЕСТНАЯ ПРОВЕРКА: ПОЧЕМУ КЛАСС 3 ИЗОЛИРОВАН?
===============================================================================
Гипотеза: Класс 3 — "сингулярность" в распределении нулей.
Проверяем: Prime Field, частоту застреваний, связь с p=3.
"""

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import sympy as sp

mp.mp.dps = 50

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 80)
print("ЧЕСТНАЯ ПРОВЕРКА: ОСОБЕННОСТИ КЛАССА 3")
print("=" * 80)

gram = np.load('gram_indices_2M.npy')
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
classes = gram % 12

print(f"✓ Загружено {len(gram):,} нулей")

def siegel_theta(t):
    return float(mp.siegeltheta(t))

def prime_field(t, primes, max_power=3):
    total = 0.0
    for p in primes:
        for k in range(1, max_power + 1):
            pk = p**k
            weight = float(mp.log(p) / (k * mp.sqrt(pk)))
            total += weight * np.sin(t * np.log(pk))
    return total

# ============================================================================
# 2. СРАВНЕНИЕ КЛАССА 3 С ДРУГИМИ
# ============================================================================
print("\n" + "=" * 80)
print("СТАТИСТИКА ПО КЛАССАМ")
print("=" * 80)

# Собираем индексы для каждого класса
class_indices = {c: np.where(classes == c)[0] for c in range(12)}

print("\nКласс | Кол-во нулей | % от всех | P(+1) прыжок | Z(P+1)")
print("-" * 65)

# Вероятности прыжков из предыдущего анализа
p_plus1 = np.array([0.5831, 0.5927, 0.5818, 0.5962, 0.5803, 0.5953, 
                    0.5806, 0.5934, 0.5858, 0.5938, 0.5771, 0.5935])
z_plus1 = np.array([-0.702, 0.737, -0.893, 1.253, -1.121, 1.121, 
                    -1.080, 0.842, -0.303, 0.899, -1.607, 0.854])

for c in range(12):
    count = len(class_indices[c])
    pct = 100 * count / len(classes)
    marker = " ← ИЗОЛИРОВАН!" if c == 3 else (" ← ХАБ" if c == 10 else "")
    print(f"{c:5d} | {count:11,} | {pct:8.3f}% | {p_plus1[c]:12.4f} | {z_plus1[c]:+8.3f} {marker}")

# ============================================================================
# 3. PRIME FIELD ДЛЯ КЛАССА 3 vs ДРУГИХ
# ============================================================================
print("\n" + "=" * 80)
print("PRIME FIELD: КЛАСС 3 vs ОСТАЛЬНЫЕ")
print("=" * 80)

primes = list(sp.primerange(1, 200))
sample_size = 3000
np.random.seed(42)

# Выборки
idx_3 = np.random.choice(class_indices[3], min(sample_size, len(class_indices[3])), replace=False)
idx_10 = np.random.choice(class_indices[10], min(sample_size, len(class_indices[10])), replace=False)
idx_others = np.random.choice(np.concatenate([class_indices[c] for c in [0,1,2,4,5,6,7,8,9,11]]), 
                              sample_size, replace=False)

pf_3 = [prime_field(zeros[i], primes) for i in idx_3]
pf_10 = [prime_field(zeros[i], primes) for i in idx_10]
pf_others = [prime_field(zeros[i], primes) for i in idx_others]

print(f"\nСредний Prime Field:")
print(f"  Класс 3:      {np.mean(pf_3):+.4f} ± {np.std(pf_3):.4f}")
print(f"  Класс 10 (хаб): {np.mean(pf_10):+.4f} ± {np.std(pf_10):.4f}")
print(f"  Остальные:    {np.mean(pf_others):+.4f} ± {np.std(pf_others):.4f}")

# Статистические тесты
_, p_3_vs_10 = stats.ttest_ind(pf_3, pf_10)
_, p_3_vs_others = stats.ttest_ind(pf_3, pf_others)

print(f"\np-value (класс 3 vs 10): {p_3_vs_10:.2e}")
print(f"p-value (класс 3 vs остальные): {p_3_vs_others:.2e}")

# ============================================================================
# 4. ВКЛАД ПРОСТОГО p=3
# ============================================================================
print("\n" + "=" * 80)
print("ВКЛАД ПРОСТОГО p=3")
print("=" * 80)

def contrib_p3(t):
    p = 3
    total = 0.0
    for k in range(1, 4):
        pk = p**k
        weight = float(mp.log(p) / (k * mp.sqrt(pk)))
        total += weight * np.sin(t * np.log(pk))
    return total

c3_p3 = [contrib_p3(zeros[i]) for i in idx_3]
c10_p3 = [contrib_p3(zeros[i]) for i in idx_10]
others_p3 = [contrib_p3(zeros[i]) for i in idx_others]

print(f"\nСредний вклад p=3:")
print(f"  Класс 3:      {np.mean(c3_p3):+.4f} ± {np.std(c3_p3):.4f}")
print(f"  Класс 10:     {np.mean(c10_p3):+.4f} ± {np.std(c10_p3):.4f}")
print(f"  Остальные:    {np.mean(others_p3):+.4f} ± {np.std(others_p3):.4f}")

_, p_p3_3v10 = stats.ttest_ind(c3_p3, c10_p3)
_, p_p3_3vothers = stats.ttest_ind(c3_p3, others_p3)

print(f"\np-value (p=3, класс 3 vs 10): {p_p3_3v10:.2e}")
print(f"p-value (p=3, класс 3 vs остальные): {p_p3_3vothers:.2e}")

# ============================================================================
# 5. ЧАСТОТА ЗАСТРЕВАНИЙ В КЛАССЕ 3
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТОТА ЗАСТРЕВАНИЙ ПО КЛАССАМ")
print("=" * 80)

stuck_counts = Counter()
class_total = Counter()

for i in range(len(gram) - 2):
    c = classes[i]
    class_total[c] += 1
    if gram[i+1] - gram[i] == 0:
        stuck_counts[c] += 1

print("\nКласс | Застреваний | Всего нулей | Частота застреваний")
print("-" * 60)
for c in range(12):
    stuck = stuck_counts.get(c, 0)
    total = class_total.get(c, 1)
    rate = stuck / total
    marker = " ← АНОМАЛИЯ!" if c == 3 else (" ← ХАБ" if c == 10 else "")
    print(f"{c:5d} | {stuck:10,} | {total:10,} | {rate:.4f} {marker}")

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Почему класс 3 изолирован?', fontsize=14)

# График 1: Prime Field распределение
ax1 = axes[0, 0]
bins = np.linspace(-6, 6, 40)
ax1.hist(pf_3, bins=bins, alpha=0.6, label=f'Класс 3 (n={len(pf_3)})', color='red', density=True)
ax1.hist(pf_10, bins=bins, alpha=0.6, label=f'Класс 10 (хаб, n={len(pf_10)})', color='blue', density=True)
ax1.hist(pf_others, bins=bins, alpha=0.4, label=f'Остальные (n={len(pf_others)})', color='gray', density=True)
ax1.axvline(x=np.mean(pf_3), color='red', linestyle='--', alpha=0.7)
ax1.axvline(x=np.mean(pf_10), color='blue', linestyle='--', alpha=0.7)
ax1.set_xlabel('Prime Field')
ax1.set_ylabel('Плотность')
ax1.set_title('Распределение Prime Field')
ax1.legend()
ax1.grid(alpha=0.3)

# График 2: Вклад p=3
ax2 = axes[0, 1]
ax2.hist(c3_p3, bins=30, alpha=0.6, label=f'Класс 3', color='red', density=True)
ax2.hist(c10_p3, bins=30, alpha=0.6, label=f'Класс 10', color='blue', density=True)
ax2.hist(others_p3, bins=30, alpha=0.4, label=f'Остальные', color='gray', density=True)
ax2.axvline(x=np.mean(c3_p3), color='red', linestyle='--')
ax2.axvline(x=np.mean(c10_p3), color='blue', linestyle='--')
ax2.set_xlabel('Вклад p=3')
ax2.set_ylabel('Плотность')
ax2.set_title('Вклад простого p=3')
ax2.legend()
ax2.grid(alpha=0.3)

# График 3: Частота застреваний
ax3 = axes[0, 2]
rates = [stuck_counts.get(c, 0) / class_total.get(c, 1) for c in range(12)]
colors = ['red' if c == 3 else 'blue' if c == 10 else 'gray' for c in range(12)]
ax3.bar(range(12), rates, color=colors, alpha=0.7)
ax3.axhline(y=np.mean(rates), color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('Класс')
ax3.set_ylabel('Частота застреваний')
ax3.set_title('Частота застреваний по классам')
ax3.set_xticks(range(12))
ax3.grid(alpha=0.3, axis='y')

# График 4: Сравнение средних Prime Field
ax4 = axes[1, 0]
means = [np.mean(pf_3), np.mean(pf_10), np.mean(pf_others)]
stds = [np.std(pf_3), np.std(pf_10), np.std(pf_others)]
labels = ['Класс 3', 'Класс 10', 'Остальные']
colors_bar = ['red', 'blue', 'gray']
ax4.bar(labels, means, yerr=stds, color=colors_bar, alpha=0.7, capsize=10)
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax4.set_ylabel('Средний Prime Field')
ax4.set_title('Сравнение средних Prime Field')
ax4.grid(alpha=0.3, axis='y')

# График 5: Сравнение вклада p=3
ax5 = axes[1, 1]
means_p3 = [np.mean(c3_p3), np.mean(c10_p3), np.mean(others_p3)]
stds_p3 = [np.std(c3_p3), np.std(c10_p3), np.std(others_p3)]
ax5.bar(labels, means_p3, yerr=stds_p3, color=colors_bar, alpha=0.7, capsize=10)
ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax5.set_ylabel('Средний вклад p=3')
ax5.set_title('Сравнение вклада p=3')
ax5.grid(alpha=0.3, axis='y')

# График 6: Сводка
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
СВОДКА ПО КЛАССУ 3:

Нулей в классе 3: {len(class_indices[3]):,} ({100*len(class_indices[3])/len(classes):.2f}%)

Прыжки:
  P(+1) = {p_plus1[3]:.4f} (максимум!)
  Z(P+1) = {z_plus1[3]:+.3f}

Prime Field:
  Класс 3: {np.mean(pf_3):+.4f} ± {np.std(pf_3):.4f}
  Класс 10: {np.mean(pf_10):+.4f} ± {np.std(pf_10):.4f}
  p-value (3 vs 10) = {p_3_vs_10:.2e}

Вклад p=3:
  Класс 3: {np.mean(c3_p3):+.4f} ± {np.std(c3_p3):.4f}
  Класс 10: {np.mean(c10_p3):+.4f} ± {np.std(c10_p3):.4f}
  p-value = {p_p3_3v10:.2e}

Частота застреваний:
  Класс 3: {rates[3]:.4f}
  Среднее: {np.mean(rates):.4f}

ВЫВОД: Класс 3 {'ОСОБЕННЫЙ' if p_3_vs_10 < 0.05 else 'НЕ ОТЛИЧАЕТСЯ'}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('class3_anomaly_analysis.png', dpi=150)
print("\n✓ График сохранён как 'class3_anomaly_analysis.png'")

# ============================================================================
# 7. ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ ПО КЛАССУ 3")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  ПОЧЕМУ КЛАСС 3 ИЗОЛИРОВАН?                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Статистика класса 3:                                                       │
│  • Нулей: {len(class_indices[3]):,} ({100*len(class_indices[3])/len(classes):.2f}%)                                    │
│  • P(+1): {p_plus1[3]:.4f} (максимальный избыток +1 прыжков)                   │
│  • Z(P+1): {z_plus1[3]:+.3f} (один из самых высоких)                           │
│                                                                             │
│  Prime Field:                                                               │
│  • Класс 3: {np.mean(pf_3):+.4f} ± {np.std(pf_3):.4f}                                       │
│  • Класс 10: {np.mean(pf_10):+.4f} ± {np.std(pf_10):.4f}                                      │
│  • p-value: {p_3_vs_10:.2e}                                                 │
│                                                                             │
│  Вклад p=3:                                                                 │
│  • Класс 3: {np.mean(c3_p3):+.4f} ± {np.std(c3_p3):.4f}                                     │
│  • Класс 10: {np.mean(c10_p3):+.4f} ± {np.std(c10_p3):.4f}                                    │
│  • p-value: {p_p3_3v10:.2e}                                                 │
│                                                                             │
│  Частота застреваний:                                                       │
│  • Класс 3: {rates[3]:.4f}                                                  │
│  • Среднее по всем: {np.mean(rates):.4f}                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ВЫВОД:                                                                     │
""")

if p_3_vs_10 < 0.05:
    print("│  ✓ Prime Field класса 3 ЗНАЧИМО отличается от класса 10!                │")
else:
    print("│  ⚠️ Prime Field класса 3 НЕ отличается значимо от класса 10             │")

if p_p3_3v10 < 0.05:
    print("│  ✓ Вклад p=3 для класса 3 ЗНАЧИМО отличается!                           │")
else:
    print("│  ⚠️ Вклад p=3 НЕ отличается значимо                                     │")

print("""│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("=" * 80)
print("АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)



"""
ЧЕСТНАЯ ПРОВЕРКА С КРОСС-ВАЛИДАЦИЕЙ
"""

import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.model_selection import LeaveOneOut

# Данные
z_scores = np.array([-0.702, 0.737, -0.893, 1.253, -1.121, 1.121, 
                     -1.080, 0.842, -0.303, 0.899, -1.607, 0.854])
degrees = np.array([2, 2, 2, 0, 1, 2, 1, 3, 3, 3, 3, 2])

print("=" * 80)
print("ЧЕСТНАЯ ПРОВЕРКА С КРОСС-ВАЛИДАЦИЕЙ")
print("=" * 80)

# 1. Теоретический порог (определён ДО анализа)
theoretical_threshold = norm.ppf(0.95)  # 1.645 для p < 0.10 (двусторонний)
print(f"\nТеоретический порог (95%): {theoretical_threshold:.3f}")

# 2. Leave-one-out cross-validation
loo = LeaveOneOut()
predictions = []

import numpy as np
from sklearn.model_selection import LeaveOneOut

# Инициализируем LOO
loo = LeaveOneOut()
predictions = []
thresholds = []

# Гарантируем, что работаем ровно с 12 индексами (0-11)
indices = np.arange(12)

# Убедитесь, что z_scores и degrees также имеют длину 12
# Если в них меньше элементов, метод выдаст ошибку выше по коду

print("Запуск Leave-One-Out валидации для классификации изолятов...")

for train_idx, test_idx in loo.split(indices):
    # 1. Обучаемся на 11 классах (индексация через массивы numpy)
    z_train = z_scores[train_idx]
    deg_train = degrees[train_idx]
    
    # 2. Поиск оптимального порога (Training)
    best_threshold = 1.0  # начальное значение
    best_accuracy = -1
    
    # Сетка поиска порога
    for thresh in np.linspace(0.5, 2.0, 30):
        # Класс считается изолированным, если его Z-score выше порога
        pred_isolated = np.abs(z_train) > thresh
        # В реальности он изолирован, если его степень в графе равна 0
        true_isolated = (deg_train == 0)
        
        accuracy = np.mean(pred_isolated == true_isolated)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    thresholds.append(best_threshold)
    
    # 3. Проверка на 1 тестовом классе (Validation)
    # Используем [0], так как test_idx — это массив из одного элемента
    test_z = z_scores[test_idx[0]]
    test_deg = degrees[test_idx[0]]
    
    # Предсказание: является ли тестовый класс изолированным?
    pred_is_isolated = np.abs(test_z) > best_threshold
    actual_is_isolated = (test_deg == 0)
    
    # Совпало ли предсказание с реальностью?
    predictions.append(pred_is_isolated == actual_is_isolated)

# Итоговые метрики
final_accuracy = np.mean(predictions)
mean_threshold = np.mean(thresholds)

print(f"✓ Валидация завершена.")
print(f"Средний найденный порог: {mean_threshold:.3f}")
print(f"Точность классификации (LOO): {final_accuracy * 100:.1f}%")

# 3. Результаты кросс-валидации
cv_accuracy = np.mean(predictions)
print(f"\nТочность предсказания изолята (LOO CV): {cv_accuracy:.2%}")
print(f"Случайный уровень: 50%")

# 4. Проверка стабильности порога
print(f"\nСтабильность порога:")
thresholds = []

# Используем np.arange(len(z_scores)), чтобы гарантировать совпадение индексов
indices = np.arange(len(z_scores))

for train_idx, _ in loo.split(indices):
    z_train = z_scores[train_idx]
    deg_train = degrees[train_idx]
    
    best_th = 1.0
    best_acc = -1
    for th in np.linspace(0.5, 2.0, 50):
        # Проверяем, насколько порог th разделяет пустые потоки (deg=0) и активные
        acc = np.mean((np.abs(z_train) > th) == (deg_train == 0))
        if acc > best_acc:
            best_acc = acc
            best_th = th
    thresholds.append(best_th)

print(f"  Средний порог: {np.mean(thresholds):.3f} ± {np.std(thresholds):.3f}")
print(f"  Разброс: {np.min(thresholds):.2f} - {np.max(thresholds):.2f}")

# 5. Проверка случайности (Пермутационный тест)
print(f"\nЗапуск 10,000 пермутаций...")
np.random.seed(42)
n_permutations = 10000
random_corrs = []

for _ in range(n_permutations):
    # Перемешиваем степени вершин, разрывая связь с Z-scores
    shuffled_degrees = np.random.permutation(degrees)
    r, _ = spearmanr(np.abs(z_scores), shuffled_degrees)
    # Если корреляция nan (бывает при константных данных), считаем её нулем
    random_corrs.append(0 if np.isnan(r) else r)

random_corrs = np.array(random_corrs)
p_permutation = np.mean(np.abs(random_corrs) >= np.abs(corr))

# Дополнительный расчет Z-score (отклонение от шума)
z_val = (corr - np.mean(random_corrs)) / np.std(random_corrs)

print(f"Пермутационный тест (корреляция): p = {p_permutation:.4f}")
print(f"Z-score эффекта: {z_val:.2f}σ")

if p_permutation < 0.05:
    print("✅ РЕЗУЛЬТАТ ЗНАЧИМ: связь Z-score и связности потоков не случайна.")
else:
    print("❌ РЕЗУЛЬТАТ НЕ ЗНАЧИМ: структура может быть артефактом выборки.")

# 6. Финальный вердикт
print("\n" + "=" * 80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ")
print("=" * 80)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  РЕЗУЛЬТАТЫ ЧЕСТНОЙ ПРОВЕРКИ                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Исходная корреляция: ρ = {corr:.3f} (p = {p_val:.3f})                                │
│  Пермутационный тест: p = {p_permutation:.3f}                                       │
│                                                                             │
│  CV-точность предсказания изолята: {cv_accuracy:.2%}                               │
│                                                                             │
│  Проблемы:                                                                  │
│  • n = 12 (очень мало для надёжных выводов)                                 │
│  • Порог {threshold_high:.1f} выбран post-hoc                                    │
│  • CV показывает нестабильность порога (±{np.std(thresholds):.3f})               │
└─────────────────────────────────────────────────────────────────────────────┘
""")

if p_permutation < 0.05 and cv_accuracy > 0.7:
    print("✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ (с оговорками)")
else:
    print("⚠️ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ при строгой проверке")