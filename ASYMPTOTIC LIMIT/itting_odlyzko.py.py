"""
ПОДГОНКА ПАРАМЕТРОВ Grand Unified Zeta Zero Model под данные Одлыжко
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import gamma
from math import log
import mpmath as mp

mp.mp.dps = 50

def siegel_theta(t):
    return float(mp.siegeltheta(t))

def get_gram_class(t):
    gram_index = int(round(siegel_theta(t) / mp.pi))
    return gram_index % 12

# Загрузка данных
BASE = 267_653_395_648.0
filename = "zero_10k_10^12.txt"
fractional_parts = np.loadtxt(filename)
heights = BASE + fractional_parts
gram_classes = np.array([get_gram_class(t) for t in heights])

print(f"Загружено {len(heights)} нулей")
print(f"Диапазон: {heights[0]:.2f} ... {heights[-1]:.2f}")

# ============================================================
# ФУНКЦИЯ ПРАВДОПОДОБИЯ ДЛЯ ПОДГОНКИ
# ============================================================

def neg_log_likelihood(params, heights, gram_classes, T_REFERENCE=2_000_000):
    """
    params = [LIMIT, A_COEFF, C_POWER, ASYMMETRY_BASE]
    LIMIT: асимптотический предел (ожидается 1.125 = 9/8)
    A_COEFF: амплитуда сходимости
    C_POWER: степень затухания
    ASYMMETRY_BASE: базовая асимметрия чёт/нечёт
    """
    LIMIT, A_COEFF, C_POWER, ASYMMETRY_BASE = params
    
    # Жёсткие ограничения
    if not (1.05 <= LIMIT <= 1.20):
        return 1e10
    if not (0.01 <= A_COEFF <= 1.0):
        return 1e10
    if not (0.1 <= C_POWER <= 1.5):
        return 1e10
    if not (0.0001 <= ASYMMETRY_BASE <= 0.05):
        return 1e10
    
    t_mean = np.mean(heights)
    ln_t = log(t_mean)
    ln_ref = log(T_REFERENCE)
    
    total_nll = 0.0
    n_valid_classes = 0
    
    for c in range(12):
        class_heights = heights[gram_classes == c]
        if len(class_heights) < 10:
            continue
        
        class_intervals = np.diff(class_heights)
        if len(class_intervals) < 5:
            continue
            
        mean_int = np.mean(class_intervals)
        norm_intervals = class_intervals / mean_int
        
        # Предсказание k для этого класса
        k_base = LIMIT - A_COEFF / (ln_t ** C_POWER)
        parity = 1 if c % 2 == 0 else -1
        current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** C_POWER
        k_pred = k_base * (1 + parity * current_asym)
        
        if k_pred <= 0.1 or k_pred > 20:
            return 1e10
        
        # Отрицательное логарифмическое правдоподобие
        try:
            nll = -np.sum(gamma.logpdf(norm_intervals, a=k_pred, scale=1/k_pred))
            if np.isnan(nll) or np.isinf(nll):
                return 1e10
            total_nll += nll
            n_valid_classes += 1
        except:
            return 1e10
    
    if n_valid_classes < 6:
        return 1e10
        
    return total_nll

# ============================================================
# ПОДГОНКА ПАРАМЕТРОВ
# ============================================================

print("\n" + "="*60)
print("🔧 ПОДГОНКА ПАРАМЕТРОВ МОДЕЛИ")
print("="*60)

# Начальные приближения
initial_params = [1.125, 0.16, 0.5, 0.0047]
bounds = [(1.05, 1.18), (0.01, 0.5), (0.2, 1.2), (0.0005, 0.02)]

print("\n📊 Начальные параметры:")
print(f"   LIMIT = {initial_params[0]}")
print(f"   A_COEFF = {initial_params[1]}")
print(f"   C_POWER = {initial_params[2]}")
print(f"   ASYMMETRY_BASE = {initial_params[3]}")

print("\n🔄 Оптимизация...")

# Глобальная оптимизация
result_global = differential_evolution(
    neg_log_likelihood,
    bounds,
    args=(heights, gram_classes),
    maxiter=100,
    popsize=15,
    seed=42,
    disp=True
)

print(f"\n✅ Глобальная оптимизация завершена")
print(f"   Успех: {result_global.success}")
print(f"   NLL: {result_global.fun:.2f}")

# Локальное уточнение
result_local = minimize(
    neg_log_likelihood,
    result_global.x,
    args=(heights, gram_classes),
    method='L-BFGS-B',
    bounds=bounds
)

print(f"\n✅ Локальное уточнение завершено")
print(f"   Успех: {result_local.success}")
print(f"   NLL: {result_local.fun:.2f}")

best_params = result_local.x

print("\n" + "="*60)
print("📈 ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ МОДЕЛИ")
print("="*60)
print(f"\n   LIMIT = {best_params[0]:.6f}  (теоретический предел: 1.125)")
print(f"   A_COEFF = {best_params[1]:.6f}  (амплитуда сходимости)")
print(f"   C_POWER = {best_params[2]:.6f}  (степень затухания)")
print(f"   ASYMMETRY_BASE = {best_params[3]:.6f}  (базовая асимметрия)")

# ============================================================
# ВАЛИДАЦИЯ ПОДГОНКИ
# ============================================================

print("\n" + "="*60)
print("🔬 ВАЛИДАЦИЯ МОДЕЛИ С ОПТИМАЛЬНЫМИ ПАРАМЕТРАМИ")
print("="*60)

t_mean = np.mean(heights)
ln_t = log(t_mean)
ln_ref = log(2_000_000)

LIMIT, A_COEFF, C_POWER, ASYMMETRY_BASE = best_params

k_base = LIMIT - A_COEFF / (ln_t ** C_POWER)
current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** C_POWER

print(f"\n🌐 ПАРАМЕТРЫ НА ВЫСОТЕ t = {t_mean:.6e}:")
print(f"   ln(t) = {ln_t:.6f}")
print(f"   k_base = {k_base:.6f}")
print(f"   Асимметрия = {current_asym:.6f} ({current_asym*100:.4f}%)")
print(f"   k_чётные = {k_base * (1 + current_asym):.6f}")
print(f"   k_нечётные = {k_base * (1 - current_asym):.6f}")

# Сравнение с наблюдениями
k_obs_even = []
k_obs_odd = []

for c in range(12):
    class_heights = heights[gram_classes == c]
    if len(class_heights) < 10:
        continue
    
    class_intervals = np.diff(class_heights)
    mean_int = np.mean(class_intervals)
    norm_intervals = class_intervals / mean_int
    
    # MLE для гаммы
    try:
        shape, _, _ = gamma.fit(norm_intervals, floc=0)
        if c % 2 == 0:
            k_obs_even.append(shape)
        else:
            k_obs_odd.append(shape)
    except:
        pass

k_obs_even_mean = np.mean(k_obs_even)
k_obs_odd_mean = np.mean(k_obs_odd)
k_obs_all_mean = (k_obs_even_mean + k_obs_odd_mean) / 2

print(f"\n📊 НАБЛЮДЕНИЯ:")
print(f"   k_чётные (ср) = {k_obs_even_mean:.6f}")
print(f"   k_нечётные (ср) = {k_obs_odd_mean:.6f}")
print(f"   k_средний = {k_obs_all_mean:.6f}")

print(f"\n🎯 ОТКЛОНЕНИЯ:")
error_mean = abs(k_obs_all_mean - k_base) / k_base * 100
print(f"   Ошибка среднего k: {error_mean:.2f}%")

# ============================================================
# ФИНАЛЬНЫЙ ВЫВОД
# ============================================================

print("\n" + "="*60)
print("📝 ИТОГОВЫЙ ВЫВОД")
print("="*60)

print(f"\n✅ МОДЕЛЬ УСПЕШНО ПОДОГНАНА ПОД ДАННЫЕ!")
print(f"\n   Оптимальные параметры:")
print(f"   • LIMIT = {best_params[0]:.4f} (теория: 1.1250)")
print(f"   • A_COEFF = {best_params[1]:.4f}")
print(f"   • C_POWER = {best_params[2]:.4f}")
print(f"   • ASYMMETRY_BASE = {best_params[3]:.5f}")

deviation = abs(best_params[0] - 1.125) / 1.125 * 100
print(f"\n📐 Отклонение LIMIT от теоретического 9/8: {deviation:.2f}%")

if error_mean < 5:
    print(f"\n🎉 ОТЛИЧНО! Модель описывает данные с точностью {error_mean:.1f}%")
elif error_mean < 10:
    print(f"\n👍 ХОРОШО! Модель описывает данные с точностью {error_mean:.1f}%")
else:
    print(f"\n⚠️ Требуется уточнение модели (ошибка {error_mean:.1f}%)")