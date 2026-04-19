"""
ЧЕСТНАЯ ПРОВЕРКА: сравнение модели с фиксированным пределом 1.125
и свободной подгонкой предела.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import gamma
from math import log
import mpmath as mp
import os

mp.mp.dps = 50

def siegel_theta(t):
    return float(mp.siegeltheta(t))

def get_gram_class(t):
    gram_index = int(round(siegel_theta(t) / mp.pi))
    return gram_index % 12

# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
print("=" * 70)
print("ЗАГРУЗКА 2 000 000 НУЛЕЙ")
print("=" * 70)

zeros = np.loadtxt('zeros_2M.txt')
N_ZEROS = min(2_000_000, len(zeros))
zeros = zeros[:N_ZEROS]
print(f"Загружено {N_ZEROS} нулей")

cache_file = 'gram_indices_2M.npy'
if os.path.exists(cache_file):
    print(f"Загрузка индексов из {cache_file}...")
    gram_indices = np.load(cache_file)
else:
    print("Вычисление индексов Грама...")
    gram_indices = np.array([int(round(siegel_theta(t) / np.pi)) for t in zeros])
    np.save(cache_file, gram_indices)

gram_classes = gram_indices % 12

# Подвыборка для ускорения
sample_idx = np.arange(0, N_ZEROS, 10)
heights_sample = zeros[sample_idx]
gram_sample = gram_classes[sample_idx]
print(f"Подвыборка: {len(heights_sample)} нулей")

# ============================================================
# ФУНКЦИЯ ПРАВДОПОДОБИЯ
# ============================================================
def neg_log_likelihood(params, heights, gram_classes, fix_limit=None):
    """
    Если fix_limit не None, то LIMIT фиксирован.
    params: если fix_limit=None -> [LIMIT, A, C, ASYM]
            если fix_limit=1.125 -> [A, C, ASYM]
    """
    if fix_limit is None:
        LIMIT, A_COEFF, C_POWER, ASYMMETRY_BASE = params
    else:
        LIMIT = fix_limit
        A_COEFF, C_POWER, ASYMMETRY_BASE = params
    
    # Ограничения
    if not (1.05 <= LIMIT <= 1.25):
        return 1e10
    if not (0.001 <= A_COEFF <= 1.0):
        return 1e10
    if not (0.1 <= C_POWER <= 2.0):
        return 1e10
    if not (0.0001 <= ASYMMETRY_BASE <= 0.05):
        return 1e10
    
    t_mean = np.mean(heights)
    ln_t = log(t_mean)
    ln_ref = log(2_000_000)
    
    total_nll = 0.0
    n_valid = 0
    
    for c in range(12):
        class_heights = heights[gram_classes == c]
        if len(class_heights) < 100:
            continue
        
        class_intervals = np.diff(class_heights)
        if len(class_intervals) < 50:
            continue
            
        mean_int = np.mean(class_intervals)
        norm_intervals = class_intervals / mean_int
        
        k_base = LIMIT - A_COEFF / (ln_t ** C_POWER)
        parity = 1 if c % 2 == 0 else -1
        current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** C_POWER
        k_pred = k_base * (1 + parity * current_asym)
        
        if k_pred <= 0.1 or k_pred > 20:
            return 1e10
        
        try:
            nll = -np.sum(gamma.logpdf(norm_intervals, a=k_pred, scale=1/k_pred))
            if np.isnan(nll) or np.isinf(nll):
                return 1e10
            total_nll += nll
            n_valid += 1
        except:
            return 1e10
    
    if n_valid < 6:
        return 1e10
    return total_nll

# ============================================================
# МОДЕЛЬ 1: СВОБОДНЫЙ ПРЕДЕЛ
# ============================================================
print("\n" + "=" * 70)
print("МОДЕЛЬ 1: СВОБОДНЫЙ ПРЕДЕЛ (4 параметра)")
print("=" * 70)

bounds_free = [(1.05, 1.25), (0.001, 0.5), (0.1, 2.0), (0.0001, 0.05)]

result_free = differential_evolution(
    lambda p: neg_log_likelihood(p, heights_sample, gram_sample, fix_limit=None),
    bounds_free,
    maxiter=50,
    popsize=10,
    seed=42,
    disp=False
)

result_free = minimize(
    lambda p: neg_log_likelihood(p, heights_sample, gram_sample, fix_limit=None),
    result_free.x,
    method='L-BFGS-B',
    bounds=bounds_free
)

params_free = result_free.x
nll_free = result_free.fun

print(f"\n🔓 СВОБОДНАЯ ПОДГОНКА:")
print(f"   LIMIT = {params_free[0]:.6f}")
print(f"   A_COEFF = {params_free[1]:.6f}")
print(f"   C_POWER = {params_free[2]:.6f}")
print(f"   ASYMMETRY_BASE = {params_free[3]:.6f}")
print(f"   NLL = {nll_free:.2f}")

# ============================================================
# МОДЕЛЬ 2: ФИКСИРОВАННЫЙ ПРЕДЕЛ 1.125
# ============================================================
print("\n" + "=" * 70)
print("МОДЕЛЬ 2: ФИКСИРОВАННЫЙ ПРЕДЕЛ 1.125 = 9/8 (3 параметра)")
print("=" * 70)

bounds_fixed = [(0.001, 0.5), (0.1, 2.0), (0.0001, 0.05)]

result_fixed = differential_evolution(
    lambda p: neg_log_likelihood(p, heights_sample, gram_sample, fix_limit=1.125),
    bounds_fixed,
    maxiter=50,
    popsize=10,
    seed=42,
    disp=False
)

result_fixed = minimize(
    lambda p: neg_log_likelihood(p, heights_sample, gram_sample, fix_limit=1.125),
    result_fixed.x,
    method='L-BFGS-B',
    bounds=bounds_fixed
)

params_fixed = result_fixed.x
nll_fixed = result_fixed.fun

print(f"\n🔒 ФИКСИРОВАННЫЙ ПРЕДЕЛ 1.125:")
print(f"   A_COEFF = {params_fixed[0]:.6f}")
print(f"   C_POWER = {params_fixed[1]:.6f}")
print(f"   ASYMMETRY_BASE = {params_fixed[2]:.6f}")
print(f"   NLL = {nll_fixed:.2f}")

# ============================================================
# СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================
print("\n" + "=" * 70)
print("📊 СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

delta_nll = nll_fixed - nll_free
print(f"\n   NLL (свободный предел):    {nll_free:.2f}")
print(f"   NLL (фиксированный 1.125): {nll_fixed:.2f}")
print(f"   Разница ΔNLL:              {delta_nll:.2f}")

# Критерий Акаике (AIC)
# AIC = 2k - 2ln(L) = 2k + 2*NLL
k_free = 4
k_fixed = 3
aic_free = 2 * k_free + 2 * nll_free
aic_fixed = 2 * k_fixed + 2 * nll_fixed
delta_aic = aic_fixed - aic_free

print(f"\n   AIC (свободный):   {aic_free:.2f}")
print(f"   AIC (фикс. 1.125): {aic_fixed:.2f}")
print(f"   ΔAIC:              {delta_aic:.2f}")

print("\n" + "=" * 70)
print("📝 ИНТЕРПРЕТАЦИЯ")
print("=" * 70)

if delta_aic < 2:
    print("\n✅✅✅ МОДЕЛЬ С ФИКСИРОВАННЫМ 1.125 НЕ ХУЖЕ!")
    print("   Данные ОТЛИЧНО описываются теорией с пределом 9/8.")
    print("   Добавление свободного предела не даёт значимого улучшения.")
elif delta_aic < 7:
    print("\n👍 МОДЕЛЬ С 1.125 ПРИЕМЛЕМА.")
    print("   Свободный предел даёт небольшое улучшение, но")
    print("   фиксированный 1.125 всё ещё хорошо описывает данные.")
else:
    print("\n⚠️ ДАННЫЕ ПРЕДПОЧИТАЮТ ДРУГОЙ ПРЕДЕЛ.")
    print(f"   Свободная подгонка даёт предел {params_free[0]:.4f},")
    print("   что значимо лучше описывает данные, чем 1.125.")

# ============================================================
# ДОПОЛНИТЕЛЬНО: ПРОВЕРКА ДРУГИХ ФИКСИРОВАННЫХ ПРЕДЕЛОВ
# ============================================================
print("\n" + "=" * 70)
print("🔬 ПРОВЕРКА ДРУГИХ ФИКСИРОВАННЫХ ПРЕДЕЛОВ")
print("=" * 70)

test_limits = [1.100, 1.125, 1.150, 1.180, 1.200]
results = []

for lim in test_limits:
    res = minimize(
        lambda p: neg_log_likelihood(p, heights_sample, gram_sample, fix_limit=lim),
        [0.16, 0.5, 0.0047],
        method='L-BFGS-B',
        bounds=bounds_fixed
    )
    results.append((lim, res.fun))
    print(f"   LIMIT = {lim:.3f} → NLL = {res.fun:.2f}")

best_lim, best_nll = min(results, key=lambda x: x[1])
print(f"\n   ЛУЧШИЙ фиксированный предел: {best_lim:.3f} (NLL = {best_nll:.2f})")

if abs(best_lim - 1.125) < 0.01:
    print("\n   🎉 1.125 — оптимальный среди проверенных!")
elif best_lim > 1.125:
    print(f"\n   📊 Данные предпочитают более высокий предел ({best_lim:.3f})")
else:
    print(f"\n   📊 Данные предпочитают более низкий предел ({best_lim:.3f})")

print("\n" + "=" * 70)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 70)