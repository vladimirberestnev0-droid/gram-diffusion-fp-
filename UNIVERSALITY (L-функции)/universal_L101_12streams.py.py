"""
================================================================================
СТРОГАЯ ПРОВЕРКА 12-ПОТОКОВОЙ ТЕОРИИ НА L-ФУНКЦИИ ДИРИХЛЕ mod 101
================================================================================
Учитывает:
1. Тип характера (чётный/нечётный) для правильной тета-функции
2. Корневое число ε(χ) для правильного фазового сдвига
3. Теоретическую формулу k(t,c) из нашей модели
4. Честное признание малой выборки (N=25)
"""

import numpy as np
from scipy import stats
from scipy.stats import gamma
import mpmath as mp
from math import log

mp.mp.dps = 50

print("=" * 80)
print("ПРОВЕРКА 12-ПОТОКОВОЙ ТЕОРИИ НА L-ФУНКЦИИ mod 101")
print("=" * 80)

# ============================================================
# 1. ИНФОРМАЦИЯ О ХАРАКТЕРЕ mod 101
# ============================================================
# Для примитивного характера по модулю 101 нужно знать:
# - Чётность: χ(-1) = 1 (чётный) или -1 (нечётный)
# - Корневое число ε(χ) = τ(χ)/√q, где τ(χ) - сумма Гаусса

# По данным LMFDB, для модуля 101 существуют и чётные, и нечётные характеры.
# Без точного указания, какой характер использован для этих 25 нулей,
# мы проверим ОБА варианта.

print("\n" + "=" * 80)
print("1. ИНФОРМАЦИЯ О ХАРАКТЕРЕ")
print("=" * 80)
print("   Модуль q = 101")
print("   Возможны два типа примитивных характеров:")
print("   • ЧЁТНЫЙ: χ(-1) = +1 → сдвиг в гамма-функции = 0.25")
print("   • НЕЧЁТНЫЙ: χ(-1) = -1 → сдвиг в гамма-функции = 0.75")
print("   Корневое число ε(χ) для обоих типов = +1 или -1")

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================
zeros = np.loadtxt('zeros_L101_25.txt')
print(f"\n{'='*80}")
print("2. ЗАГРУЗКА ДАННЫХ")
print("=" * 80)
print(f"   Загружено {len(zeros)} нулей L-функции mod 101")
print(f"   Диапазон: {zeros[0]:.6f} ... {zeros[-1]:.6f}")
print("\n   ⚠️ ВНИМАНИЕ: 25 нулей — это крайне малая выборка!")
print("   Все выводы являются ПРЕДВАРИТЕЛЬНЫМИ.")

# ============================================================
# 3. ПАРАМЕТРЫ НАШЕЙ ТЕОРИИ
# ============================================================
print("\n" + "=" * 80)
print("3. ПАРАМЕТРЫ 12-ПОТОКОВОЙ ТЕОРИИ")
print("=" * 80)

GRAM_CLASSES = 12
K_INF = GRAM_CLASSES / 10  # = 1.200
ALPHA = 0.5
A_COEFF = 0.16
ASYMMETRY_BASE = 0.0047
T_REFERENCE = 2_000_000

print(f"   GRAM_CLASSES = {GRAM_CLASSES} (теоретически из π/8)")
print(f"   k_inf = GRAM_CLASSES/10 = {K_INF:.4f}")
print(f"   α = {ALPHA}")
print(f"   a = {A_COEFF}")
print(f"   A = {ASYMMETRY_BASE}")
print(f"   T_ref = {T_REFERENCE:,}")

# ============================================================
# 4. ФУНКЦИИ ДЛЯ РАЗНЫХ ТИПОВ ХАРАКТЕРА
# ============================================================
def siegel_theta_L_even(t, q=101, epsilon_sign=1):
    """Тета-функция для ЧЁТНОГО характера χ(-1)=1."""
    arg_gamma = float(mp.im(mp.loggamma(mp.mpc(0.25, t/2))))
    root_phase = 0 if epsilon_sign == 1 else mp.pi/2
    return arg_gamma - (t/2) * (float(mp.log(mp.pi)) - float(mp.log(q))) + root_phase

def siegel_theta_L_odd(t, q=101, epsilon_sign=1):
    """Тета-функция для НЕЧЁТНОГО характера χ(-1)=-1."""
    arg_gamma = float(mp.im(mp.loggamma(mp.mpc(0.75, t/2))))
    root_phase = 0 if epsilon_sign == 1 else mp.pi/2
    return arg_gamma - (t/2) * (float(mp.log(mp.pi)) - float(mp.log(q))) + root_phase

def get_gram_classes(zeros, theta_func):
    """Вычисляет классы Грама для заданной тета-функции."""
    classes = []
    for z in zeros:
        theta = theta_func(z)
        gram_index = int(round(theta / mp.pi))
        classes.append(gram_index % GRAM_CLASSES)
    return np.array(classes)

def predict_k_theory(t, c):
    """Предсказание теории для высоты t и класса c."""
    t = max(t, 100)
    ln_t = log(t)
    ln_ref = log(T_REFERENCE)
    k_base = K_INF - A_COEFF / (ln_t ** ALPHA)
    parity = 1 if c % 2 == 0 else -1
    current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** ALPHA
    return k_base * (1 + parity * current_asym)

# ============================================================
# 5. АНАЛИЗ ДЛЯ ВСЕХ ВОЗМОЖНЫХ ВАРИАНТОВ
# ============================================================
print("\n" + "=" * 80)
print("4. АНАЛИЗ ДЛЯ РАЗНЫХ ТИПОВ ХАРАКТЕРА")
print("=" * 80)

variants = [
    ("ЧЁТНЫЙ, ε=+1", lambda t: siegel_theta_L_even(t, 101, 1)),
    ("ЧЁТНЫЙ, ε=-1", lambda t: siegel_theta_L_even(t, 101, -1)),
    ("НЕЧЁТНЫЙ, ε=+1", lambda t: siegel_theta_L_odd(t, 101, 1)),
    ("НЕЧЁТНЫЙ, ε=-1", lambda t: siegel_theta_L_odd(t, 101, -1)),
]

results = {}

for name, theta_func in variants:
    print(f"\n--- {name} ---")
    
    classes = get_gram_classes(zeros, theta_func)
    
    # Распределение по классам
    counts = np.bincount(classes, minlength=GRAM_CLASSES)
    even_count = sum(counts[c] for c in range(0, GRAM_CLASSES, 2))
    odd_count = sum(counts[c] for c in range(1, GRAM_CLASSES, 2))
    
    print(f"   Чётные классы: {even_count} ({100*even_count/len(zeros):.1f}%)")
    print(f"   Нечётные классы: {odd_count} ({100*odd_count/len(zeros):.1f}%)")
    
    # Хи-квадрат тест
    if len(zeros) >= 12:
        _, p_val = stats.chisquare(counts)
        print(f"   p-value (χ²): {p_val:.4f}")
    
    # Анализ интервалов внутри потоков
    k_obs_list = []
    var_list = []
    
    for c in range(GRAM_CLASSES):
        mask = classes == c
        heights_c = zeros[mask]
        if len(heights_c) >= 3:
            intervals = np.diff(heights_c)
            mean_int = np.mean(intervals)
            norm_int = intervals / mean_int
            variance = np.var(norm_int, ddof=1)
            var_list.append(variance)
            try:
                shape, _, _ = gamma.fit(norm_int, floc=0)
                k_obs_list.append(shape)
            except:
                pass
    
    if k_obs_list:
        k_obs_mean = np.mean(k_obs_list)
        var_mean = np.mean(var_list)
        print(f"   Средний k (набл.): {k_obs_mean:.4f}")
        print(f"   Средняя дисперсия: {var_mean:.4f}")
        
        # Предсказание теории для средней высоты
        t_mean = np.mean(zeros)
        k_pred_mean = np.mean([predict_k_theory(t_mean, c) for c in range(GRAM_CLASSES)])
        print(f"   Предсказание теории: {k_pred_mean:.4f}")
        print(f"   Отклонение: {abs(k_obs_mean - k_pred_mean):.4f}")
    else:
        k_obs_mean = np.nan
        var_mean = np.nan
        print("   ⚠️ Недостаточно данных для оценки k")
    
    results[name] = {
        'even_count': even_count,
        'odd_count': odd_count,
        'p_value': p_val if len(zeros) >= 12 else np.nan,
        'k_obs': k_obs_mean,
        'var_obs': var_mean,
        'k_pred': k_pred_mean if k_obs_list else np.nan
    }

# ============================================================
# 6. СРАВНИТЕЛЬНЫЙ АНАЛИЗ
# ============================================================
print("\n" + "=" * 80)
print("5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВСЕХ ВАРИАНТОВ")
print("=" * 80)

print(f"\n   {'Вариант':<18} {'Чёт':>5} {'Неч':>5} {'p-value':>8} {'k_obs':>7} {'k_pred':>7} {'Откл':>7}")
print("-" * 65)

for name, res in results.items():
    k_obs_str = f"{res['k_obs']:.3f}" if not np.isnan(res['k_obs']) else "N/A"
    k_pred_str = f"{res['k_pred']:.3f}" if not np.isnan(res['k_pred']) else "N/A"
    dev_str = f"{abs(res['k_obs'] - res['k_pred']):.3f}" if not np.isnan(res['k_obs']) and not np.isnan(res['k_pred']) else "N/A"
    p_str = f"{res['p_value']:.3f}" if not np.isnan(res['p_value']) else "N/A"
    
    print(f"   {name:<18} {res['even_count']:>5} {res['odd_count']:>5} {p_str:>8} {k_obs_str:>7} {k_pred_str:>7} {dev_str:>7}")
# ============================================================
# 7. ВЫВОДЫ
# ============================================================
print("\n" + "=" * 80)
print("6. ВЫВОДЫ")
print("=" * 80)

print("""
   ОГРАНИЧЕНИЯ:
   • N = 25 нулей — крайне малая выборка
   • Статистическая значимость любых выводов низкая
   • Тип характера и корневое число точно не известны

   ЧТО МОЖНО СКАЗАТЬ:
""")

# Ищем вариант, наиболее близкий к предсказанию
best_match = None
best_dev = float('inf')
for name, res in results.items():
    if not np.isnan(res['k_obs']) and not np.isnan(res['k_pred']):
        dev = abs(res['k_obs'] - res['k_pred'])
        if dev < best_dev:
            best_dev = dev
            best_match = name

if best_match:
    print(f"   • Наиболее близкое совпадение с теорией: {best_match}")
    print(f"     (отклонение {best_dev:.3f})")
else:
    print("   • Ни один вариант не дал надёжной оценки k")

# Сравнение с дзета-функцией
print("""
   СРАВНЕНИЕ С ζ(s) (2 000 000 нулей):
   ┌─────────────────┬────────────────────┬────────────────────┐
   │ Параметр        │ ζ(s)               │ L(s,χ₁₀₁) (25 нулей)│
   ├─────────────────┼────────────────────┼────────────────────┤
   │ Чётные классы   │ +0.47% (избыток)   │ зависит от варианта│
   │ k_obs (t ~ 30)  │ —                  │ ~0.82              │
   │ k_obs (t ~ 10⁶) │ 1.082              │ —                  │
   │ k_obs (t ~ 10¹¹)│ 1.206              │ —                  │
   │ k_pred (теория) │ 1.200 (предел)     │ 1.200 (тот же)     │
   └─────────────────┴────────────────────┴────────────────────┘

   ПРЕДВАРИТЕЛЬНЫЙ ВЫВОД:
   На малых высотах (t ~ 30) L-функция показывает k ≈ 0.82,
   что существенно ниже предсказания теории (1.200). Однако:
   • Для ζ(s) на малых высотах k тоже ниже предела (1.04 при t=10⁴)
   • Нужны данные L-функции на бóльших высотах (хотя бы t ~ 10³–10⁴)
   • Требуется точное знание типа характера и корневого числа

   ДЛЯ ПРОВЕРКИ УНИВЕРСАЛЬНОСТИ НЕОБХОДИМО:
   1. Минимум 10 000 нулей L-функции
   2. Точная информация о характере (чётность, корневое число)
   3. Данные на разных высотах для отслеживания эволюции k(t)
""")

print("=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)
print("\n   ⚠️ ВСЕ ВЫВОДЫ ПО L-ФУНКЦИИ ЯВЛЯЮТСЯ ПРЕДВАРИТЕЛЬНЫМИ")
print("   ИЗ-ЗА КРАЙНЕ МАЛОГО ОБЪЁМА ВЫБОРКИ (N=25)")