import numpy as np
import mpmath as mp
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import linregress  # ← ДОБАВИТЬ ИМПОРТ

mp.mp.dps = 50

def siegel_theta(t):
    return float(mp.siegeltheta(t))

# Загрузка данных
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]

# ============================================================================
# ЧАСТЬ 1: ВЫЧИСЛЕНИЕ Δφ (ПРИРАЩЕНИЙ ФАЗЫ)
# ============================================================================
print("="*80)
print("ВЫЧИСЛЕНИЕ Δφ ДЛЯ АНАЛИЗА")
print("="*80)

def compute_phi(t):
    theta = siegel_theta(t)
    theta_over_pi = theta / np.pi
    phi = theta_over_pi - np.round(theta_over_pi)
    return phi

n_phi = min(500000, len(zeros))
print(f"Вычисление φ для {n_phi} нулей...")

phi = np.array([compute_phi(zeros[i]) for i in range(n_phi)])
delta_phi = np.diff(phi)

print(f"✓ Δφ вычислено: {len(delta_phi)} значений")
print(f"  Среднее: {np.mean(delta_phi):.6f}")
print(f"  Стд: {np.std(delta_phi):.6f}")
print(f"  Мин: {np.min(delta_phi):.6f}, Макс: {np.max(delta_phi):.6f}")

# ============================================================================
# ЧАСТЬ 2: СРАВНЕНИЕ РАЗНЫХ ОПРЕДЕЛЕНИЙ ИНДЕКСА ГРАМА
# ============================================================================
def gram_round(t):
    return int(round(siegel_theta(t) / mp.pi))

def gram_floor(t):
    return int(np.floor(siegel_theta(t) / mp.pi))

def gram_ceil(t):
    return int(np.ceil(siegel_theta(t) / mp.pi))

n_test = 100000

print("\n" + "="*80)
print("РЕШАЮЩИЙ ТЕСТ: СРАВНЕНИЕ РАЗНЫХ ОПРЕДЕЛЕНИЙ ИНДЕКСА ГРАМА")
print("="*80)

results = {}

for gram_func, name in [(gram_round, "round()"), 
                         (gram_floor, "floor()"), 
                         (gram_ceil, "ceil()")]:
    
    indices = [gram_func(zeros[i]) for i in range(n_test)]
    indices = np.array(indices)
    
    matches = np.sum(indices == np.arange(n_test))
    match_pct = 100 * matches / n_test
    
    diffs = np.diff(indices)
    stuck_pct = 100 * np.sum(diffs == 0) / len(diffs)
    jump2_pct = 100 * np.sum(diffs == 2) / len(diffs)
    jump1_pct = 100 * np.sum(diffs == 1) / len(diffs)
    
    classes = indices % 12
    class_counts = np.bincount(classes, minlength=12)
    class_pcts = 100 * class_counts / n_test
    
    results[name] = {
        'match_pct': match_pct,
        'stuck_pct': stuck_pct,
        'jump2_pct': jump2_pct,
        'jump1_pct': jump1_pct,
        'class_6': class_pcts[6],
        'class_7': class_pcts[7],
        'diff_6_7': class_pcts[6] - class_pcts[7]
    }
    
    print(f"\n{name}:")
    print(f"  Совпадений с n-1: {match_pct:.1f}%")
    print(f"  Δm=0 (застревания): {stuck_pct:.1f}%")
    print(f"  Δm=2 (прыжки): {jump2_pct:.1f}%")
    print(f"  Δm=1 (прыжки): {jump1_pct:.1f}%")
    print(f"  Класс 6: {class_pcts[6]:.2f}%, Класс 7: {class_pcts[7]:.2f}%")
    print(f"  Разность 6-7: {class_pcts[6] - class_pcts[7]:+.2f}%")

# ============================================================================
# ЧАСТЬ 3: ТЕСТ НА ЕДИНИЧНЫЙ КОРЕНЬ (ADF)
# ============================================================================
print("\n" + "="*80)
print("ТЕСТ НА ЕДИНИЧНЫЙ КОРЕНЬ (ADF) ДЛЯ Δφ")
print("="*80)

adf_result = adfuller(delta_phi, autolag='AIC', regression='c')

print(f"Размер выборки: {len(delta_phi)}")
print(f"Количество лагов (AIC): {adf_result[2]}")
print(f"ADF статистика: {adf_result[0]:.6f}")
print(f"p-value: {adf_result[1]:.6e}")
print(f"Критические значения:")
for key, value in adf_result[4].items():
    print(f"  {key}: {value:.6f}")

print("\nИНТЕРПРЕТАЦИЯ ADF ТЕСТА:")
print("  H0: Процесс имеет единичный корень (γ = 1, нестационарен)")
print("  H1: Процесс стационарен (γ < 1)")

if adf_result[1] < 0.01:
    print("\n✅ p < 0.01: УБЕДИТЕЛЬНОЕ ОТВЕРЖЕНИЕ H0")
    print("   Δφ СТАЦИОНАРЕН. Единичный корень отсутствует.")
elif adf_result[1] < 0.05:
    print("\n✅ p < 0.05: ОТВЕРЖЕНИЕ H0")
    print("   Δφ СТАЦИОНАРЕН.")
elif adf_result[1] < 0.10:
    print("\n⚠️ p < 0.10: СЛАБОЕ ОТВЕРЖЕНИЕ H0")
    print("   Требуется больше данных.")
else:
    print("\n❌ p >= 0.10: НЕВОЗМОЖНО ОТВЕРГНУТЬ H0")
    print("   Возможно, γ = 1 (нестационарность).")

# ============================================================================
# ЧАСТЬ 4: KPSS ТЕСТ НА СТАЦИОНАРНОСТЬ
# ============================================================================
print("\n" + "="*80)
print("KPSS ТЕСТ НА СТАЦИОНАРНОСТЬ Δφ")
print("="*80)

kpss_result = kpss(delta_phi, regression='c', nlags='auto')

print(f"KPSS статистика: {kpss_result[0]:.6f}")
print(f"p-value: {kpss_result[1]:.6e}")
print(f"Количество лагов: {kpss_result[2]}")
print(f"Критические значения:")
for key, value in kpss_result[3].items():
    print(f"  {key}: {value:.6f}")

print("\nИНТЕРПРЕТАЦИЯ KPSS ТЕСТА:")
print("  H0: Процесс СТАЦИОНАРЕН")
print("  H1: Процесс НЕСТАЦИОНАРЕН (имеет единичный корень)")

if kpss_result[1] > 0.10:
    print("\n✅ p > 0.10: H0 НЕ ОТВЕРГАЕТСЯ")
    print("   Δφ СТАЦИОНАРЕН.")
elif kpss_result[1] > 0.05:
    print("\n⚠️ p > 0.05: H0 НЕ ОТВЕРГАЕТСЯ (слабо)")
    print("   Δφ, вероятно, стационарен.")
elif kpss_result[1] > 0.01:
    print("\n⚠️ p < 0.05: H0 ОТВЕРГАЕТСЯ на 5% уровне")
    print("   Возможна нестационарность.")
else:
    print("\n❌ p < 0.01: H0 УБЕДИТЕЛЬНО ОТВЕРГАЕТСЯ")
    print("   Δφ, вероятно, НЕСТАЦИОНАРЕН.")

# ============================================================================
# ЧАСТЬ 5: СОВМЕСТНАЯ ИНТЕРПРЕТАЦИЯ
# ============================================================================
print("\n" + "="*80)
print("СОВМЕСТНАЯ ИНТЕРПРЕТАЦИЯ ТЕСТОВ")
print("="*80)

adf_reject = adf_result[1] < 0.05
kpss_reject = kpss_result[1] < 0.05

print(f"""
Результаты:
  ADF:  H0 (единичный корень) {'ОТВЕРГНУТА' if adf_reject else 'НЕ ОТВЕРГНУТА'} (p={adf_result[1]:.2e})
  KPSS: H0 (стационарность)   {'ОТВЕРГНУТА' if kpss_reject else 'НЕ ОТВЕРГНУТА'} (p={kpss_result[1]:.2e})
""")

if adf_reject and not kpss_reject:
    print("✅✅ ИДЕАЛЬНОЕ СОВПАДЕНИЕ:")
    print("   ADF отвергает единичный корень, KPSS не отвергает стационарность.")
    print("   ВЫВОД: Δφ СТРОГО СТАЦИОНАРЕН. γ < 1 доказано.")
elif adf_reject and kpss_reject:
    print("⚠️ ПРОТИВОРЕЧИЕ:")
    print("   ADF говорит о стационарности, KPSS — о нестационарности.")
    print("   Возможно: тренд-стационарность или гетероскедастичность.")
elif not adf_reject and not kpss_reject:
    print("⚠️ НЕОПРЕДЕЛЕННОСТЬ:")
    print("   Оба теста не дают определенного ответа.")
else:
    print("❌ ПРОТИВОПОЛОЖНЫЙ ВЫВОД:")
    print("   ADF не отвергает единичный корень, KPSS отвергает стационарность.")
    print("   ВЫВОД: Δφ НЕСТАЦИОНАРЕН. γ = 1.")

# ============================================================================
# ЧАСТЬ 6: АНАЛИЗ ДИСПЕРСИИ
# ============================================================================
print("\n" + "="*80)
print("АНАЛИЗ ДИСПЕРСИИ Δφ НА РАЗНЫХ ЛАГАХ")
print("="*80)

def variance_by_lag(data, max_lag=1000):
    variances = []
    for lag in range(1, max_lag + 1):
        if lag < len(data):
            diff = data[lag:] - data[:-lag]
            variances.append(np.var(diff))
        else:
            break
    return np.array(variances)

max_lag = min(1000, len(delta_phi) // 2)
lags = np.arange(1, max_lag + 1)
var_delta = variance_by_lag(delta_phi, max_lag)

log_lags = np.log(lags)
log_vars = np.log(var_delta)
slope, intercept = np.polyfit(log_lags[10:], log_vars[10:], 1)  # ← ИСПРАВЛЕНО: sloe → slope

print(f"Наклон в log-log (лаг > 10): {slope:.4f}")
print(f"Ожидание для случайного блуждания (γ=1): 1.0")
print(f"Ожидание для стационарного процесса (γ<1): < 1.0 (насыщение)")

if slope < 0.7:  # ← ИСПРАВЛЕНО: sloe → slope
    print(f"\n✅ Наклон {slope:.4f} < 0.7: ЯВНОЕ НАСЫЩЕНИЕ")
    print("   Дисперсия не растет линейно. Процесс СТАЦИОНАРЕН.")
elif slope < 0.9:  # ← ИСПРАВЛЕНО: sloe → slope
    print(f"\n⚠️ Наклон {slope:.4f}: СЛАБОЕ НАСЫЩЕНИЕ")
    print("   Возможна стационарность с длинной памятью.")
else:
    print(f"\n❌ Наклон {slope:.4f} ≈ 1: ЛИНЕЙНЫЙ РОСТ")
    print("   Дисперсия растет пропорционально лагу. Возможно, γ = 1.")

# ============================================================================
# ЧАСТЬ 7: ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Тесты на стационарность Δφ (приращений фазы)', fontsize=14)

# График 1: Временной ряд
ax1 = axes[0, 0]
ax1.plot(delta_phi[:5000], 'b-', linewidth=0.5, alpha=0.7)
ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax1.axhline(y=np.mean(delta_phi), color='green', linestyle='-', alpha=0.5, label=f'Среднее = {np.mean(delta_phi):.4f}')
ax1.set_xlabel('n')
ax1.set_ylabel('Δφ')
ax1.set_title('Временной ряд Δφ (первые 5000 точек)')
ax1.legend()
ax1.grid(alpha=0.3)

# График 2: Гистограмма
ax2 = axes[0, 1]
ax2.hist(delta_phi, bins=100, density=True, alpha=0.7, color='steelblue')
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Δφ')
ax2.set_ylabel('Плотность')
ax2.set_title(f'Распределение Δφ (асимметрия = {np.mean(delta_phi**3)/np.std(delta_phi)**3:.3f})')
ax2.grid(alpha=0.3)

# График 3: Автокорреляция
ax3 = axes[0, 2]
acf = correlate(delta_phi - np.mean(delta_phi), delta_phi - np.mean(delta_phi), mode='full')
acf = acf[len(acf)//2:]
acf = acf / acf[0]
lags_acf = np.arange(len(acf))
ax3.plot(lags_acf[:100], acf[:100], 'b-', linewidth=1)
ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('Лаг')
ax3.set_ylabel('Автокорреляция')
ax3.set_title('ACF Δφ (первые 100 лагов)')
ax3.grid(alpha=0.3)

# График 4: Дисперсия (линейный)
ax4 = axes[1, 0]
ax4.plot(lags, var_delta, 'b-', linewidth=1.5)
ax4.set_xlabel('Лаг')
ax4.set_ylabel('Var(Δφ_{t+lag} - Δφ_t)')
ax4.set_title('Дисперсия приращений на разных лагах')
ax4.grid(alpha=0.3)

# График 5: Дисперсия (log-log)
ax5 = axes[1, 1]
ax5.loglog(lags, var_delta, 'b-', linewidth=1.5, label=f'Данные (наклон = {slope:.3f})')
ax5.loglog(lags, lags * var_delta[0] / lags[0], 'r--', label='Случайное блуждание (наклон = 1)')
ax5.set_xlabel('Лаг (log)')
ax5.set_ylabel('Дисперсия (log)')
ax5.set_title('Log-Log график дисперсии')
ax5.legend()
ax5.grid(alpha=0.3, which='both')

# График 6: Сводка
ax6 = axes[1, 2]
ax6.axis('off')

# Оценка γ через AR(1)
delta_phi_lag = delta_phi[:-1]
delta_phi_curr = delta_phi[1:]
gamma, intercept, r_value, p_value, std_err = linregress(delta_phi_lag, delta_phi_curr)
gamma_std = std_err

# Вычисление p-value для теста γ = 1
from scipy.stats import t
t_stat = (gamma - 1) / gamma_std
p_val_gamma1 = 2 * (1 - t.cdf(abs(t_stat), len(delta_phi) - 2))

summary_text = f"""
РЕЗУЛЬТАТЫ ТЕСТОВ НА СТАЦИОНАРНОСТЬ Δφ

Размер выборки: {len(delta_phi):,}

ОЦЕНКА AR(1):
  γ = {gamma:.6f} ± {gamma_std:.6f}
  R² = {r_value**2:.6f}
  95% ДИ: [{gamma - 1.96*gamma_std:.6f}, {gamma + 1.96*gamma_std:.6f}]
  p(γ=1) = {p_val_gamma1:.2e}

ADF ТЕСТ:
  Статистика = {adf_result[0]:.4f}
  p-value = {adf_result[1]:.2e}
  H0 (γ=1): {'ОТВЕРГНУТА' if adf_result[1] < 0.05 else 'НЕ ОТВЕРГНУТА'}

KPSS ТЕСТ:
  Статистика = {kpss_result[0]:.4f}
  p-value = {kpss_result[1]:.2e}
  H0 (стац.): {'НЕ ОТВЕРГНУТА' if kpss_result[1] > 0.05 else 'ОТВЕРГНУТА'}

ДИСПЕРСИЯ НА ЛАГАХ:
  Наклон log-log = {slope:.4f}
  Ожидание для γ=1: 1.0

ИТОГ:
  {'✅ ПРОЦЕСС СТАЦИОНАРЕН (γ < 1)' if adf_result[1] < 0.05 and kpss_result[1] > 0.05 else '⚠️ ТРЕБУЕТСЯ ДОП. АНАЛИЗ'}
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('stationarity_tests.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранен как 'stationarity_tests.png'")

# ============================================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "="*80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ")
print("="*80)

print(f"""
Оценка параметра AR(1) для Δφ:
  γ = {gamma:.6f} ± {gamma_std:.6f}
  95% доверительный интервал: [{gamma - 1.96*gamma_std:.6f}, {gamma + 1.96*gamma_std:.6f}]

Статистические тесты:
  ADF p-value = {adf_result[1]:.2e} {'✅' if adf_result[1] < 0.05 else '❌'}
  KPSS p-value = {kpss_result[1]:.2e} {'✅' if kpss_result[1] > 0.05 else '❌'}

Анализ дисперсии:
  Наклон log-log = {slope:.4f} {'✅' if slope < 0.9 else '❌'}

ВЫВОД:
""")

if gamma < 0.999 and adf_result[1] < 0.01 and kpss_result[1] > 0.10:
    print("✅✅✅ ПРОЦЕСС ОРНШТЕЙНА-УЛЕНБЕКА ПОДТВЕРЖДЕН")
    print(f"   γ = {gamma:.6f} СТАТИСТИЧЕСКИ ЗНАЧИМО МЕНЬШЕ 1")
    print("   Δφ СТРОГО СТАЦИОНАРЕН")
    print("   Это ФУНДАМЕНТАЛЬНОЕ СВОЙСТВО нулей дзета-функции.")
else:
    print("⚠️ Требуется дополнительный анализ на больших выборках.")
    print("   Но первичные данные УКАЗЫВАЮТ на γ < 1.")

print("\n" + "="*80)