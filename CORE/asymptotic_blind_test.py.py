"""
================================================================================
СТРОГАЯ ПРОВЕРКА ГИПОТЕЗЫ k_inf = 12/10 = 1.200
Без подгонки, без свободных параметров
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma
import mpmath as mp
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

mp.mp.dps = 50

# ============================================================
# ФИКСИРОВАННЫЕ ПАРАМЕТРЫ ТЕОРИИ (НЕ ПОДГОНЯЮТСЯ!)
# ============================================================
K_INF = 1.200                    # Предел 12/10
ALPHA = 0.5                      # Степень логарифмического затухания
A_COEFF = 0.16                   # Амплитуда сходимости
ASYMMETRY_BASE = 0.0047          # Базовая асимметрия
T_REFERENCE = 2_000_000          # Опорная высота

print("=" * 80)
print("СТРОГАЯ ПРОВЕРКА ГИПОТЕЗЫ k_inf = 12/10 = 1.200")
print("=" * 80)
print(f"\n🔒 ФИКСИРОВАННЫЕ ПАРАМЕТРЫ (без подгонки):")
print(f"   k_inf = {K_INF:.4f} = 12/10")
print(f"   alpha = {ALPHA:.2f}")
print(f"   a = {A_COEFF:.4f}")
print(f"   A = {ASYMMETRY_BASE:.4f}")
print(f"   T_ref = {T_REFERENCE:,}")

# ============================================================
# ФУНКЦИИ
# ============================================================
def siegel_theta(t):
    return float(mp.siegeltheta(t))

def get_gram_class(t):
    gram_index = int(round(siegel_theta(t) / mp.pi))
    return gram_index % 12

def predict_k_theory(t, c):
    """
    СТРОГОЕ предсказание теории для высоты t и класса c.
    Никаких свободных параметров!
    """
    t = max(t, 100)
    ln_t = np.log(t)
    ln_ref = np.log(T_REFERENCE)
    
    # Базовая сходимость
    k_base = K_INF - A_COEFF / (ln_t ** ALPHA)
    
    # Чётностная модуляция
    parity = 1 if c % 2 == 0 else -1
    current_asym = ASYMMETRY_BASE * (ln_ref / ln_t) ** ALPHA
    
    return k_base * (1 + parity * current_asym)

# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
print("\n" + "=" * 80)
print("1. ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

# Данные 2M нулей
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
print(f"   ✓ Загружено {len(zeros):,} нулей ζ(s)")
print(f"     t ∈ [{zeros[0]:.2f}, {zeros[-1]:.2f}]")

# Данные Одлыжко
try:
    fractional = np.loadtxt('zero_10k_10^12.txt')
    BASE = 267_653_395_648.0
    odlyzko = BASE + fractional
    print(f"   ✓ Загружено {len(odlyzko):,} нулей Одлыжко")
    print(f"     t ∈ [{odlyzko[0]:.2f}, {odlyzko[-1]:.2f}]")
    HAS_ODLYZKO = True
except:
    print("   ⚠️ Данные Одлыжко не найдены")
    HAS_ODLYZKO = False

# ============================================================
# 2. ВЫЧИСЛЕНИЕ КЛАССОВ
# ============================================================
print("\n" + "=" * 80)
print("2. ВЫЧИСЛЕНИЕ КЛАССОВ ГРАМА")
print("=" * 80)

cache = 'gram_indices_2M.npy'
if os.path.exists(cache):
    gram_indices = np.load(cache)
    print(f"   ✓ Загружено из кэша")
else:
    print("   ⏳ Вычисление...")
    gram_indices = np.array([get_gram_class(z) for z in tqdm(zeros)])
    np.save(cache, gram_indices)

gram_classes = gram_indices % 12

if HAS_ODLYZKO:
    print("   ⏳ Вычисление для Одлыжко...")
    odlyzko_classes = np.array([get_gram_class(z) for z in tqdm(odlyzko)]) % 12

# ============================================================
# 3. ПРОВЕРКА НА 2M НУЛЯХ (t ~ 10^6)
# ============================================================
print("\n" + "=" * 80)
print("3. ПРОВЕРКА НА 2 000 000 НУЛЯХ (t ~ 10^6)")
print("=" * 80)

t_mean = np.mean(zeros)
print(f"\n   Средняя высота: {t_mean:.2e}")

# Вычисляем наблюдаемые k
k_obs_even = []
k_obs_odd = []

for c in range(12):
    mask = gram_classes == c
    heights_c = zeros[mask]
    if len(heights_c) > 1000:
        intervals = np.diff(heights_c)
        norm_int = intervals / np.mean(intervals)
        try:
            shape, _, _ = gamma.fit(norm_int, floc=0)
            if c % 2 == 0:
                k_obs_even.append(shape)
            else:
                k_obs_odd.append(shape)
        except:
            pass

k_obs_mean = np.mean(k_obs_even + k_obs_odd)
k_obs_even_mean = np.mean(k_obs_even)
k_obs_odd_mean = np.mean(k_obs_odd)

# Предсказания теории
k_pred_even = predict_k_theory(t_mean, 0)
k_pred_odd = predict_k_theory(t_mean, 1)
k_pred_mean = (k_pred_even + k_pred_odd) / 2

print(f"\n   📊 НАБЛЮДЕНИЯ:")
print(f"      k средний:     {k_obs_mean:.4f}")
print(f"      k чётные:      {k_obs_even_mean:.4f}")
print(f"      k нечётные:    {k_obs_odd_mean:.4f}")
print(f"      Разрыв:        {k_obs_even_mean - k_obs_odd_mean:.4f}")

print(f"\n   🔮 ПРЕДСКАЗАНИЯ ТЕОРИИ:")
print(f"      k средний:     {k_pred_mean:.4f}")
print(f"      k чётные:      {k_pred_even:.4f}")
print(f"      k нечётные:    {k_pred_odd:.4f}")
print(f"      Разрыв:        {k_pred_even - k_pred_odd:.4f}")

# Ошибки
err_mean = abs(k_obs_mean - k_pred_mean) / k_pred_mean * 100
err_even = abs(k_obs_even_mean - k_pred_even) / k_pred_even * 100
err_odd = abs(k_obs_odd_mean - k_pred_odd) / k_pred_odd * 100
err_gap = abs((k_obs_even_mean - k_obs_odd_mean) - (k_pred_even - k_pred_odd)) / (k_pred_even - k_pred_odd) * 100

print(f"\n   🎯 ОШИБКИ:")
print(f"      Среднее:  {err_mean:.2f}%")
print(f"      Чётные:   {err_even:.2f}%")
print(f"      Нечётные: {err_odd:.2f}%")
print(f"      Разрыв:   {err_gap:.2f}%")

max_err = max(err_mean, err_even, err_odd, err_gap)
if max_err < 5:
    print(f"\n   ✅✅✅ ОТЛИЧНО! Все ошибки < 5%")
elif max_err < 10:
    print(f"\n   👍 ХОРОШО! Все ошибки < 10%")
else:
    print(f"\n   ⚠️ Есть расхождения > 10%")

# ============================================================
# 4. ПРОВЕРКА НА ДАННЫХ ОДЛЫЖКО (t ~ 10^11)
# ============================================================
if HAS_ODLYZKO:
    print("\n" + "=" * 80)
    print("4. ПРОВЕРКА НА ДАННЫХ ОДЛЫЖКО (t ~ 2.68e11)")
    print("=" * 80)
    
    t_odlyzko = np.mean(odlyzko)
    print(f"\n   Средняя высота: {t_odlyzko:.2e}")
    
    # Наблюдения
    k_obs_even_od = []
    k_obs_odd_od = []
    
    for c in range(12):
        mask = odlyzko_classes == c
        heights_c = odlyzko[mask]
        if len(heights_c) > 10:
            intervals = np.diff(heights_c)
            norm_int = intervals / np.mean(intervals)
            try:
                shape, _, _ = gamma.fit(norm_int, floc=0)
                if c % 2 == 0:
                    k_obs_even_od.append(shape)
                else:
                    k_obs_odd_od.append(shape)
            except:
                pass
    
    k_obs_mean_od = np.mean(k_obs_even_od + k_obs_odd_od)
    k_obs_even_mean_od = np.mean(k_obs_even_od) if k_obs_even_od else np.nan
    k_obs_odd_mean_od = np.mean(k_obs_odd_od) if k_obs_odd_od else np.nan
    
    # Предсказания
    k_pred_even_od = predict_k_theory(t_odlyzko, 0)
    k_pred_odd_od = predict_k_theory(t_odlyzko, 1)
    k_pred_mean_od = (k_pred_even_od + k_pred_odd_od) / 2
    
    print(f"\n   📊 НАБЛЮДЕНИЯ:")
    print(f"      k средний:     {k_obs_mean_od:.4f}")
    print(f"      k чётные:      {k_obs_even_mean_od:.4f}")
    print(f"      k нечётные:    {k_obs_odd_mean_od:.4f}")
    
    print(f"\n   🔮 ПРЕДСКАЗАНИЯ ТЕОРИИ:")
    print(f"      k средний:     {k_pred_mean_od:.4f}")
    print(f"      k чётные:      {k_pred_even_od:.4f}")
    print(f"      k нечётные:    {k_pred_odd_od:.4f}")
    
    # Ошибки
    err_mean_od = abs(k_obs_mean_od - k_pred_mean_od) / k_pred_mean_od * 100
    
    print(f"\n   🎯 ОШИБКА СРЕДНЕГО: {err_mean_od:.2f}%")
    
    # Сравнение с альтернативной гипотезой
    k_pred_125 = 1.125 - 0.16 / (np.log(t_odlyzko) ** 0.5)
    err_125 = abs(k_obs_mean_od - k_pred_125) / k_pred_125 * 100
    
    print(f"\n   📊 СРАВНЕНИЕ С ГИПОТЕЗОЙ 9/8 = 1.125:")
    print(f"      Предсказание 9/8: {k_pred_125:.4f}")
    print(f"      Ошибка 9/8:       {err_125:.2f}%")
    
    if err_mean_od < 5:
        print(f"\n   ✅✅✅ БЛЕСТЯЩЕ! Ошибка < 5% на t = 10^11!")
    elif err_mean_od < 10:
        print(f"\n   👍 ОЧЕНЬ ХОРОШО! Ошибка < 10%")
    else:
        print(f"\n   ⚠️ Расхождение > 10%")
    
    if err_mean_od < err_125:
        print(f"\n   🏆 Гипотеза 12/10 = 1.200 ТОЧНЕЕ, чем 9/8 = 1.125")
        print(f"      (ошибка {err_mean_od:.1f}% против {err_125:.1f}%)")

# ============================================================
# 5. ИТОГОВЫЙ ВЕРДИКТ
# ============================================================
print("\n" + "=" * 80)
print("5. ИТОГОВЫЙ ВЕРДИКТ")
print("=" * 80)

print(f"""
   ГИПОТЕЗА: k_inf = 12/10 = 1.200
   ПАРАМЕТРЫ: α = 0.5, a = 0.16, A = 0.0047 (фиксированы)

   РЕЗУЛЬТАТЫ ПРОВЕРКИ:
   ┌─────────────────┬──────────────┬──────────────┐
   │ Высота t        │ Наблюдаемое k│ Предсказание │
   ├─────────────────┼──────────────┼──────────────┤
   │ t ~ 1.0e6       │   {k_obs_mean:.4f}     │   {k_pred_mean:.4f}     │
""")

if HAS_ODLYZKO:
    print(f"   │ t ~ 2.7e11      │   {k_obs_mean_od:.4f}     │   {k_pred_mean_od:.4f}     │")

print("   └─────────────────┴──────────────┴──────────────┘")

if HAS_ODLYZKO:
    if err_mean < 10 and err_mean_od < 10:
        print("\n   ✅✅✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА НА ДВУХ НЕЗАВИСИМЫХ ДИАПАЗОНАХ!")
        print("   Теория с k_inf = 12/10 = 1.200 работает!")
    else:
        print("\n   📊 Требуется уточнение параметров или больше данных")
else:
    if err_mean < 5:
        print("\n   ✅ Гипотеза подтверждена на t ~ 10^6")
        print("      Для полной проверки нужны данные Одлыжко")

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)

# ============================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Сравнение на t ~ 10^6
ax1 = axes[0]
categories = ['Среднее', 'Чётные', 'Нечётные']
obs_1 = [k_obs_mean, k_obs_even_mean, k_obs_odd_mean]
pred_1 = [k_pred_mean, k_pred_even, k_pred_odd]

x = np.arange(3)
width = 0.35
ax1.bar(x - width/2, obs_1, width, label='Наблюдения', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, pred_1, width, label='Теория (k_inf=1.200)', color='orange', alpha=0.8)
ax1.axhline(y=1.200, color='green', linestyle='--', label='Предел 12/10')
ax1.set_xticks(x)
ax1.set_xticklabels(categories)
ax1.set_ylabel('Параметр k')
ax1.set_title(f't ~ 1.0e6\nОшибка среднего: {err_mean:.1f}%')
ax1.legend()
ax1.grid(alpha=0.3, axis='y')

# График 2: Сравнение на t ~ 10^11
if HAS_ODLYZKO:
    ax2 = axes[1]
    ax2.bar(x - width/2, [k_obs_mean_od, k_obs_even_mean_od, k_obs_odd_mean_od], 
            width, label='Наблюдения', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, [k_pred_mean_od, k_pred_even_od, k_pred_odd_od], 
            width, label='Теория (k_inf=1.200)', color='orange', alpha=0.8)
    ax2.axhline(y=1.200, color='green', linestyle='--', label='Предел 12/10')
    ax2.axhline(y=1.125, color='red', linestyle=':', label='Предел 9/8')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Параметр k')
    ax2.set_title(f't ~ 2.7e11\nОшибка 12/10: {err_mean_od:.1f}% | Ошибка 9/8: {err_125:.1f}%')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')
else:
    ax2 = axes[1]
    ax2.text(0.5, 0.5, 'Данные Одлыжко не найдены\n(нужен файл zero_10k_10^12.txt)', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Данные отсутствуют')

plt.tight_layout()
plt.savefig('strict_theory_test.png', dpi=150, bbox_inches='tight')
print("\n📈 График сохранён как 'strict_theory_test.png'")
plt.show()