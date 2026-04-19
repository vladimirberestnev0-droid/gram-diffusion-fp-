import numpy as np
import mpmath as mp
from scipy import stats

mp.mp.dps = 50

# === 1. Загрузка 25 нулей L-функции mod 101 ===
zeros = np.loadtxt('zeros_L101_25.txt')
print(f"Загружено {len(zeros)} нулей L-функции mod 101")

# === 2. Функция тета для L-функции (ИСПРАВЛЕНО) ===
def siegel_theta_L101(t):
    q = 101  # кондуктор
    # arg гамма-функции через мнимую часть loggamma
    arg_gamma = float(mp.im(mp.loggamma(mp.mpc(0.25, t/2))))
    return arg_gamma - (t/2) * (float(mp.log(mp.pi)) - float(mp.log(q)))

def get_gram_index(gamma):
    return int(round(siegel_theta_L101(gamma) / mp.pi))

# === 3. Вычисление индексов Грама ===
print("\nВычисление индексов Грама...")
gram_indices = np.array([get_gram_index(g) for g in zeros])
residue_class = gram_indices % 12

# === 4. Распределение по классам ===
counts = np.bincount(residue_class, minlength=12)
total = len(zeros)

print("\nРаспределение по 12 классам (m mod 12):")
print("Класс | Кол-во | Процент")
print("-" * 25)
for c in range(12):
    pct = 100 * counts[c] / total
    print(f"{c:5d} | {counts[c]:6d} | {pct:6.1f}%")

# Хи-квадрат тест
if total >= 12:
    _, p_value = stats.chisquare(counts)
    print(f"\np-value (хи-квадрат): {p_value:.4f}")

# === 5. Анализ интервалов ВНУТРИ КАЖДОГО КЛАССА ===
print("\nАнализ внутри классов (интервалы между нулями одного потока):")
print("Класс | N  | Дисперсия | k_гамма")
print("-" * 40)

all_variances = []
all_shapes = []

for c in range(12):
    mask = residue_class == c
    gamma_c = zeros[mask]
    n = len(gamma_c)
    
    if n >= 2:
        intervals = np.diff(gamma_c)
        mean_int = np.mean(intervals)
        norm_int = intervals / mean_int
        
        variance = np.var(norm_int)
        all_variances.append(variance)
        
        try:
            shape, loc, scale = stats.gamma.fit(norm_int, floc=0)
            all_shapes.append(shape)
        except:
            shape = np.nan
        
        print(f"{c:5d} | {n:2d} | {variance:.4f}    | {shape:.4f}")
    else:
        print(f"{c:5d} | {n:2d} | ---       | ---")

if all_variances:
    print(f"\nСредняя дисперсия: {np.mean(all_variances):.4f}")
    print(f"Средний k гамма:    {np.mean(all_shapes):.4f}")
    print("\nОжидаемые значения:")
    print("  Реальные нули ζ(s): дисперсия ~0.515, k ~1.08")
    print("  Случайный шум:      дисперсия ~1.000, k ~1.00")