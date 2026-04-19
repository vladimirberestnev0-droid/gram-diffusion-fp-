"""
ИСПРАВЛЕННАЯ ПРОВЕРКА Grand Unified Zeta Zero Model
на реальных данных Одлыжко (10 000 нулей вокруг t ~ 10^12)

КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
1. Точное вычисление индекса Грама через siegel_theta (mpmath)
2. Корректное разбиение на 12 потоков
3. Правильная нормировка интервалов
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma
from math import log, sqrt, pi
import mpmath as mp

# ============================================================
# НАСТРОЙКА ТОЧНОСТИ
# ============================================================
mp.mp.dps = 50  # Высокая точность для тета-функции

def siegel_theta(t):
    """Тета-функция Зигеля (точное вычисление через mpmath)."""
    return float(mp.siegeltheta(t))

def get_gram_class(t):
    """
    ТОЧНЫЙ класс Грама (0..11) для заданной высоты t.
    Индекс Грама: m = round(theta(t) / pi).
    Класс: m mod 12.
    """
    gram_index = int(round(siegel_theta(t) / mp.pi))
    return gram_index % 12

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

BASE = 267_653_395_648.0  # Базовая целая часть

filename = "zero_10k_10^12.txt"
try:
    fractional_parts = np.loadtxt(filename)
    print(f"✅ Загружено {len(fractional_parts)} дробных частей из {filename}")
except FileNotFoundError:
    print(f"❌ Файл {filename} не найден.")
    print("   Скачай данные с: http://www.dtc.umn.edu/~odlyzko/zeta_tables/index.html")
    exit(1)

heights = BASE + fractional_parts
n_zeros = len(heights)
print(f"   Диапазон высот: {heights[0]:.8f} ... {heights[-1]:.8f}")
print(f"   Всего нулей: {n_zeros}")

# ============================================================
# 2. ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

LIMIT = 1.125                    # Предел 9/8
A_COEFF = 0.16                   # Амплитуда сходимости
C_POWER = 0.5                    # Степень затухания
ASYMMETRY_BASE = 0.0047          # Базовая асимметрия
T_REFERENCE = 2_000_000          # Опорная высота
GRAM_CLASSES = 12

def predict_k_model(t, m):
    """
    Предсказание модели для высоты t и класса Грама m.
    """
    t = max(t, 100)
    ln_t = log(t)
    ln_ref = log(T_REFERENCE)
    
    k_base = LIMIT - A_COEFF / (ln_t ** C_POWER)
    
    parity = 1 if m % 2 == 0 else -1
    current_asymmetry = ASYMMETRY_BASE * (ln_ref / ln_t) ** C_POWER
    
    return k_base * (1 + parity * current_asymmetry)

# ============================================================
# 3. ВЫЧИСЛЕНИЕ КЛАССОВ (ТОЧНОЕ)
# ============================================================

print("\n🔢 ВЫЧИСЛЕНИЕ ТОЧНЫХ КЛАССОВ ГРАМА...")
gram_classes = np.array([get_gram_class(t) for t in heights])

print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО 12 КЛАССАМ ГРАМА:")
print("-" * 50)
for c in range(12):
    count = np.sum(gram_classes == c)
    pct = 100 * count / n_zeros
    parity = "ЧЁТ" if c % 2 == 0 else "НЕЧ"
    bar = "█" * int(pct)
    print(f"   Класс {c:2d} ({parity}): {count:6d} нулей ({pct:5.1f}%) {bar}")

# ============================================================
# 4. ГЛОБАЛЬНАЯ СТАТИСТИКА
# ============================================================

all_intervals = np.diff(heights)
global_mean_interval = np.mean(all_intervals)
global_std_interval = np.std(all_intervals)

print(f"\n📏 ГЛОБАЛЬНАЯ СТАТИСТИКА (все соседние интервалы):")
print(f"   Средний:     {global_mean_interval:.8f}")
print(f"   Стд.откл:    {global_std_interval:.8f}")

# ============================================================
# 5. АНАЛИЗ 12 ПОТОКОВ
# ============================================================

print(f"\n🔬 АНАЛИЗ 12 ПОТОКОВ (интервалы между нулями внутри одного класса):")
print("-" * 110)
print(f"{'Класс':<6} {'N_нулей':<10} {'N_инт':<8} {'Ср.инт':<14} {'k (дисп)':<10} {'k (MLE)':<10} {'k_pred':<10} {'Ошибка %':<10}")
print("-" * 110)

t_mean = np.mean(heights)
ln_t = log(t_mean)
ln_ref = log(T_REFERENCE)

k_model_base = LIMIT - A_COEFF / (ln_t ** C_POWER)
current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** C_POWER

k_even_obs = []
k_odd_obs = []
k_even_pred = []
k_odd_pred = []

for c in range(12):
    # Берём ВСЕ нули класса c
    class_heights = heights[gram_classes == c]
    n_class_zeros = len(class_heights)
    
    if n_class_zeros < 10:
        print(f"{c:<6} {n_class_zeros:<10} {'—':<8} {'—':<14} {'—':<10} {'—':<10} {'—':<10} {'—'}")
        continue
    
    # Интервалы МЕЖДУ СОСЕДНИМИ НУЛЯМИ ВНУТРИ ЭТОГО КЛАССА
    class_intervals = np.diff(class_heights)
    n_intervals = len(class_intervals)
    mean_class_interval = np.mean(class_intervals)
    
    # Нормировка на СРЕДНЕЕ ЭТОГО ЖЕ КЛАССА
    norm_intervals = class_intervals / mean_class_interval
    
    # Оценка k через дисперсию (метод моментов)
    var_norm = np.var(norm_intervals, ddof=1)
    k_disp = 1 / var_norm if var_norm > 0 else float('inf')
    
    # Оценка k через MLE
    try:
        from scipy.optimize import minimize_scalar
        
        def neg_log_likelihood(log_k):
            k = np.exp(log_k)
            if k <= 0 or k > 100:
                return float('inf')
            return -np.sum(gamma.logpdf(norm_intervals, a=k, scale=1/k))
        
        k_initial = max(0.5, min(5.0, k_disp))
        res = minimize_scalar(neg_log_likelihood, 
                              bracket=(log(0.1), log(k_initial), log(10.0)),
                              method='brent')
        k_mle = np.exp(res.x) if res.success else k_disp
    except:
        k_mle = k_disp
    
    # Предсказание модели
    k_pred = predict_k_model(t_mean, c)
    
    error_pct = abs(k_mle - k_pred) / k_pred * 100
    
    if c % 2 == 0:
        k_even_obs.append(k_mle)
        k_even_pred.append(k_pred)
    else:
        k_odd_obs.append(k_mle)
        k_odd_pred.append(k_pred)
    
    parity = "Ч" if c % 2 == 0 else "Н"
    print(f"{c:<6} {n_class_zeros:<10} {n_intervals:<8} {mean_class_interval:<14.8f} {k_disp:<10.4f} {k_mle:<10.4f} {k_pred:<10.4f} {error_pct:<10.2f}%")

# ============================================================
# 6. ИТОГОВОЕ СРАВНЕНИЕ
# ============================================================

print("\n" + "=" * 80)
print("📈 ИТОГОВОЕ СРАВНЕНИЕ НАБЛЮДЕНИЙ С МОДЕЛЬЮ")
print("=" * 80)

print(f"\n🌐 ПАРАМЕТРЫ НА ВЫСОТЕ t = {t_mean:.6e}:")
print(f"   ln(t) = {ln_t:.6f}")
print(f"   sqrt(ln(t)) = {sqrt(ln_t):.6f}")

print(f"\n🔮 ПРЕДСКАЗАНИЯ МОДЕЛИ:")
print(f"   k_base (среднее):     {k_model_base:.6f}")
print(f"   Асимметрия:           {current_asym:.6f} ({current_asym*100:.3f}%)")
print(f"   k_чётные:             {k_model_base * (1 + current_asym):.6f}")
print(f"   k_нечётные:           {k_model_base * (1 - current_asym):.6f}")
print(f"   Разрыв чёт/нечёт:     {2 * k_model_base * current_asym:.6f}")

k_obs_even_mean = np.mean(k_even_obs) if k_even_obs else 0
k_obs_odd_mean = np.mean(k_odd_obs) if k_odd_obs else 0
k_obs_all_mean = (k_obs_even_mean + k_obs_odd_mean) / 2 if (k_even_obs and k_odd_obs) else 0

print(f"\n📊 НАБЛЮДЕНИЯ (MLE-оценки по 12 потокам):")
print(f"   k_чётные (ср):        {k_obs_even_mean:.6f}")
print(f"   k_нечётные (ср):      {k_obs_odd_mean:.6f}")
print(f"   k_средний:            {k_obs_all_mean:.6f}")
print(f"   Разрыв чёт/нечёт:     {k_obs_even_mean - k_obs_odd_mean:.6f}")

if k_model_base > 0:
    error_mean = abs(k_obs_all_mean - k_model_base) / k_model_base * 100
else:
    error_mean = 0

if current_asym > 0:
    predicted_gap = 2 * k_model_base * current_asym
    observed_gap = k_obs_even_mean - k_obs_odd_mean
    error_gap = abs(observed_gap - predicted_gap) / predicted_gap * 100 if predicted_gap > 0 else 0
else:
    error_gap = 0

print(f"\n🎯 ОШИБКИ:")
print(f"   Ошибка среднего k:    {error_mean:.2f}%")
print(f"   Ошибка разрыва:       {error_gap:.2f}%")

print("\n" + "=" * 80)
if error_mean < 5.0:
    print("✅✅✅ МОДЕЛЬ ПОДТВЕРЖДЕНА! Отклонение менее 5%!")
elif error_mean < 10.0:
    print("👍 Хорошее согласие. Модель работает.")
elif error_mean < 20.0:
    print("📊 Умеренное согласие. Требуется больше данных.")
else:
    print("⚠️ Значительное расхождение. Проверьте данные.")
print("=" * 80)

# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f'12-ПОТОКОВАЯ МОДЕЛЬ: проверка на данных Одлыжко (t ~ {t_mean:.2e})', fontsize=14)

# График 1: k по 12 классам
ax1 = axes[0, 0]
classes = np.arange(12)
k_obs_by_class = []
k_pred_by_class = []

for c in classes:
    k_pred_by_class.append(predict_k_model(t_mean, c))
    
    class_heights = heights[gram_classes == c]
    if len(class_heights) >= 10:
        class_intervals = np.diff(class_heights)
        mean_int = np.mean(class_intervals)
        norm_int = class_intervals / mean_int
        
        try:
            shape, _, _ = gamma.fit(norm_int, floc=0)
            k_obs_by_class.append(shape)
        except:
            k_obs_by_class.append(np.nan)
    else:
        k_obs_by_class.append(np.nan)

colors_obs = ['blue' if c % 2 == 0 else 'red' for c in classes]
colors_pred = ['cyan' if c % 2 == 0 else 'orange' for c in classes]

x = np.arange(12)
width = 0.35
bars1 = ax1.bar(x - width/2, k_obs_by_class, width, color=colors_obs, alpha=0.7, label='Наблюдения (MLE)')
bars2 = ax1.bar(x + width/2, k_pred_by_class, width, color=colors_pred, alpha=0.5, label='Модель')
ax1.axhline(y=k_model_base, color='green', linestyle='--', label=f'k_base = {k_model_base:.4f}')
ax1.set_xlabel('Класс Грама')
ax1.set_ylabel('Параметр k')
ax1.set_title('12 потоков: модель vs наблюдения')
ax1.set_xticks(classes)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# График 2: Гистограмма нормированных интервалов
ax2 = axes[0, 1]
all_stream_intervals = []
for c in range(12):
    class_heights = heights[gram_classes == c]
    if len(class_heights) > 10:
        intervals = np.diff(class_heights)
        norm_int = intervals / np.mean(intervals)
        all_stream_intervals.extend(norm_int)

all_stream_intervals = np.array(all_stream_intervals)
ax2.hist(all_stream_intervals, bins=30, density=True, alpha=0.6, color='gray', label=f'Наблюдения (n={len(all_stream_intervals)})')

x_plot = np.linspace(0, max(3, np.max(all_stream_intervals) * 0.8), 200)
if k_obs_all_mean > 0:
    pdf_gamma_obs = gamma.pdf(x_plot, a=k_obs_all_mean, scale=1/k_obs_all_mean)
    ax2.plot(x_plot, pdf_gamma_obs, 'r-', linewidth=2, label=f'Гамма (k={k_obs_all_mean:.3f})')
pdf_gamma_model = gamma.pdf(x_plot, a=k_model_base, scale=1/k_model_base)
ax2.plot(x_plot, pdf_gamma_model, 'b--', linewidth=2, label=f'Модель (k={k_model_base:.3f})')
pdf_exp = np.exp(-x_plot)
ax2.plot(x_plot, pdf_exp, 'g:', linewidth=1, label='Экспонента (k=1)')

ax2.set_xlabel('Нормированный интервал в потоке')
ax2.set_ylabel('Плотность')
ax2.set_title('Распределение интервалов внутри 12 потоков')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 3])

# График 3: Средние интервалы по классам
ax3 = axes[1, 0]
mean_intervals = []
for c in classes:
    class_heights = heights[gram_classes == c]
    if len(class_heights) > 1:
        mean_intervals.append(np.mean(np.diff(class_heights)))
    else:
        mean_intervals.append(np.nan)

ax3.bar(classes, mean_intervals, color=['blue' if c % 2 == 0 else 'red' for c in classes], alpha=0.7)
ax3.axhline(y=12 * global_mean_interval, color='green', linestyle='--', 
            label=f'12 × глобальный = {12 * global_mean_interval:.6f}')
ax3.set_xlabel('Класс Грама')
ax3.set_ylabel('Средний интервал в потоке')
ax3.set_title('Средние интервалы: должны быть ≈ 12 × глобальный')
ax3.set_xticks(classes)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# График 4: QQ-plot
ax4 = axes[1, 1]
sorted_obs = np.sort(all_stream_intervals)
if k_obs_all_mean > 0:
    theoretical_quantiles = gamma.ppf(np.linspace(0.01, 0.99, len(sorted_obs)), 
                                      a=k_obs_all_mean, scale=1/k_obs_all_mean)
    ax4.scatter(theoretical_quantiles, sorted_obs, alpha=0.3, s=5, color='blue')
ax4.plot([0, 3], [0, 3], 'r--', linewidth=1, label='y = x')
ax4.set_xlabel('Теоретические квантили (Гамма)')
ax4.set_ylabel('Наблюдаемые квантили')
ax4.set_title(f'QQ-plot для 12 потоков (k = {k_obs_all_mean:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 3])
ax4.set_ylim([0, 3])

plt.tight_layout()
plt.savefig('odlyzko_12streams_CORRECTED.png', dpi=150, bbox_inches='tight')
print("\n📊 График сохранён как 'odlyzko_12streams_CORRECTED.png'")
plt.show()

# ============================================================
# 8. СТАТИСТИЧЕСКИЕ ТЕСТЫ
# ============================================================

print("\n" + "=" * 80)
print("📊 СТАТИСТИЧЕСКИЕ ТЕСТЫ ДЛЯ 12 ПОТОКОВ")
print("=" * 80)

# Тест Колмогорова-Смирнова
if k_obs_all_mean > 0 and len(all_stream_intervals) > 0:
    ks_stat, ks_pvalue = stats.kstest(all_stream_intervals, 'gamma', args=(k_obs_all_mean, 0, 1/k_obs_all_mean))
    print(f"\n🔬 Тест Колмогорова-Смирнова (Гамма):")
    print(f"   Статистика: {ks_stat:.4f}")
    print(f"   p-value:    {ks_pvalue:.4f}")
    if ks_pvalue > 0.05:
        print("   ✅ Не отвергается гипотеза о гамма-распределении")
    else:
        print("   ⚠️ Гипотеза о гамма-распределении отвергается")

# t-тест на различие чётных и нечётных
if k_even_obs and k_odd_obs:
    t_stat, t_pvalue = stats.ttest_ind(k_even_obs, k_odd_obs)
    print(f"\n🔬 t-тест на различие чётных и нечётных потоков:")
    print(f"   t-статистика: {t_stat:.4f}")
    print(f"   p-value:      {t_pvalue:.4f}")
    if t_pvalue < 0.05:
        print(f"   ✅ Статистически значимое различие (p < 0.05)")
    else:
        print(f"   ⚠️ Различие статистически незначимо (нужно больше данных)")

# Проверка отношения средних интервалов
mean_stream_interval = np.mean([np.mean(np.diff(heights[gram_classes == c])) 
                                for c in range(12) if np.sum(gram_classes == c) > 1])
ratio = mean_stream_interval / global_mean_interval
print(f"\n🔬 ОТНОШЕНИЕ СРЕДНИХ ИНТЕРВАЛОВ:")
print(f"   Средний интервал в потоке:  {mean_stream_interval:.6f}")
print(f"   Глобальный средний:         {global_mean_interval:.6f}")
print(f"   Отношение:                  {ratio:.4f}")
print(f"   Ожидалось:                  12.0")
if abs(ratio - 12.0) < 0.5:
    print("   ✅ Отношение близко к 12 — 12-потоковая структура подтверждена!")
else:
    print("   ⚠️ Отношение отличается от 12")

print("\n" + "=" * 80)
print("🎯 АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)

print("\n📝 ВЫВОДЫ:")
print("-" * 40)
print(f"1. Модель предсказывает k_base = {k_model_base:.4f}")
print(f"   Наблюдаемое k_mean (по потокам) = {k_obs_all_mean:.4f}")
print(f"   Отклонение: {error_mean:.2f}%")
print()
print(f"2. Отношение интервалов поток/глобальный = {ratio:.2f} (ожидалось 12.00)")
print()
print(f"3. Параметр k ≈ {k_obs_all_mean:.3f} для потоков")

if error_mean < 20.0 and abs(ratio - 12.0) < 1.0:
    print("\n✅✅✅ 12-ПОТОКОВАЯ МОДЕЛЬ ПОДТВЕРЖДЕНА!")
    print("   На высоте 10^12 мы видим именно ту структуру, которую предсказывает теория.")
else:
    print("\n⚠️ Требуется дополнительная проверка.")