"""
===============================================================================
ПОСТРОЕНИЕ НЕПРЕРЫВНОГО ПРЕДЕЛА ГРАММ-ДИФФУЗИИ
===============================================================================
Часть 1: Эмпирический дрейф μ(PF)
Часть 2: Оценка волатильности σ
Часть 3: Симуляция СДУ и сравнение с реальными данными
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import norm, ks_2samp, shapiro, pearsonr
import mpmath as mp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

mp.mp.dps = 50

# ============================================================================
# ЗАГРУЗКА ДАННЫХ И ВЫЧИСЛЕНИЕ НЕОБХОДИМЫХ ВЕЛИЧИН
# ============================================================================
print("=" * 80)
print("ЗАГРУЗКА ДАННЫХ И ПОДГОТОВКА")
print("=" * 80)

zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
gram = np.load('gram_indices_2M.npy')[:2_000_000]

def siegel_theta(t):
    return float(mp.siegeltheta(t))

# Вычисляем Δ и Prime Field для всех нулей
print("Вычисление Δ и Prime Field...")

deltas = []
prime_fields = []
diff_grams = []

# Простые числа для Prime Field
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def prime_field(t):
    total = 0.0
    for p in primes:
        for k in range(1, 4):
            pk = p**k
            weight = float(mp.log(p) / (k * mp.sqrt(pk)))
            total += weight * np.sin(t * np.log(pk))
    return total

for i in tqdm(range(len(zeros)), desc="Обработка нулей"):
    t = zeros[i]
    m = gram[i]
    theta = siegel_theta(t)
    delta = theta / np.pi - m
    pf = prime_field(t)
    
    deltas.append(delta)
    prime_fields.append(pf)
    
    if i < len(zeros) - 1:
        diff_grams.append(gram[i+1] - gram[i])

deltas = np.array(deltas)
prime_fields = np.array(prime_fields)
diff_grams = np.array(diff_grams)

print(f"✓ Обработано {len(deltas):,} нулей")
print(f"  Δ ∈ [{deltas.min():.4f}, {deltas.max():.4f}]")
print(f"  PF ∈ [{prime_fields.min():.4f}, {prime_fields.max():.4f}]")

# ============================================================================
# ЧАСТЬ 1: ПОСТРОЕНИЕ ЭМПИРИЧЕСКОГО ДРЕЙФА μ(PF)
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 1: ПОСТРОЕНИЕ ЭМПИРИЧЕСКОГО ДРЕЙФА μ(PF)")
print("=" * 80)

# Разбиваем на бины по Prime Field (а не по Δ, чтобы получить функцию от PF)
n_bins = 40
pf_bins = np.linspace(np.percentile(prime_fields, 1), np.percentile(prime_fields, 99), n_bins)

pf_centers = []
dm_means = []
dm_stds = []
prob_stuck = []
prob_jump = []
counts = []

for i in range(len(pf_bins) - 1):
    mask = (prime_fields[:-1] >= pf_bins[i]) & (prime_fields[:-1] < pf_bins[i+1])
    n = np.sum(mask)
    
    if n > 100:
        pf_center = (pf_bins[i] + pf_bins[i+1]) / 2
        pf_centers.append(pf_center)
        counts.append(n)
        
        dm_mean = np.mean(diff_grams[mask])
        dm_means.append(dm_mean)
        dm_stds.append(np.std(diff_grams[mask]))
        
        prob_stuck.append(np.mean(diff_grams[mask] == 0))
        prob_jump.append(np.mean(diff_grams[mask] == 2))

pf_centers = np.array(pf_centers)
dm_means = np.array(dm_means)
counts = np.array(counts)

# Подгонка сигмоидальной функции для дрейфа
def sigmoid(x, a, b, c, d):
    """a: амплитуда, b: крутизна, c: смещение по x, d: смещение по y"""
    return a / (1 + np.exp(-b * (x - c))) + d

# Начальные приближения
p0 = [-1.0, 0.5, 0.0, 1.0]  # dm убывает от 2 до 0 с ростом PF

try:
    popt, pcov = curve_fit(sigmoid, pf_centers, dm_means, p0=p0, maxfev=5000, 
                           sigma=1/np.sqrt(counts))
    a, b, c, d = popt
    
    # Качество подгонки
    dm_pred = sigmoid(pf_centers, *popt)
    r2 = 1 - np.sum((dm_means - dm_pred)**2) / np.sum((dm_means - np.mean(dm_means))**2)
    
    print(f"\n✅ Сигмоидальная подгонка:")
    print(f"   μ(PF) = {a:.4f} / (1 + exp(-{b:.4f}·(PF - {c:.4f}))) + {d:.4f}")
    print(f"   R² = {r2:.6f}")
    
    if r2 > 0.9:
        print(f"   ✅ ОТЛИЧНО: R² > 0.9, дрейф восстановлен!")
    elif r2 > 0.7:
        print(f"   👍 ХОРОШО: R² > 0.7, зависимость подтверждена")
    else:
        print(f"   ⚠️ R² = {r2:.4f}, требуется больше данных")
        
except Exception as e:
    print(f"\n⚠️ Подгонка не удалась: {e}")
    popt = None
    r2 = None

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Построение непрерывного предела Грамм-диффузии', fontsize=14)

ax1 = axes[0, 0]
ax1.errorbar(pf_centers, dm_means, yerr=dm_stds/np.sqrt(counts), fmt='o', 
             markersize=4, alpha=0.7, capsize=2, label='Эмпирические')
if popt is not None:
    pf_smooth = np.linspace(pf_centers.min(), pf_centers.max(), 200)
    dm_smooth = sigmoid(pf_smooth, *popt)
    ax1.plot(pf_smooth, dm_smooth, 'r-', linewidth=2, label=f'Подгонка (R²={r2:.4f})')
ax1.set_xlabel('Prime Field')
ax1.set_ylabel('⟨Δm⟩')
ax1.set_title('Эмпирический дрейф μ(PF)')
ax1.legend()
ax1.grid(alpha=0.3)

# График вероятностей
ax2 = axes[0, 1]
ax2.plot(pf_centers, prob_stuck, 'ro-', label='P(Δm=0)', markersize=4, alpha=0.7)
ax2.plot(pf_centers, prob_jump, 'bo-', label='P(Δm=2)', markersize=4, alpha=0.7)
ax2.set_xlabel('Prime Field')
ax2.set_ylabel('Вероятность')
ax2.set_title('Условные вероятности переходов')
ax2.legend()
ax2.grid(alpha=0.3)

# ============================================================================
# ЧАСТЬ 2: ОЦЕНКА ВОЛАТИЛЬНОСТИ σ
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 2: ОЦЕНКА ВОЛАТИЛЬНОСТИ σ")
print("=" * 80)

if popt is not None:
    # Вычисляем предсказанные значения дрейфа для каждого нуля
    pf_for_residuals = prime_fields[:-1]
    dm_pred_all = sigmoid(pf_for_residuals, *popt)
    
    # residuals = наблюдаемое - предсказанное
    residuals = diff_grams - dm_pred_all
    
    # Оценка σ
    sigma = np.std(residuals)
    
    print(f"\nОценка волатильности: σ = {sigma:.6f}")
    
    # Тест на нормальность
    # Используем подвыборку 5000 для теста Шапиро-Уилка
    sample_size = min(5000, len(residuals))
    residuals_sample = np.random.choice(residuals, sample_size, replace=False)
    shapiro_stat, shapiro_p = shapiro(residuals_sample)
    
    print(f"\nТест Шапиро-Уилка на нормальность residuals:")
    print(f"  W = {shapiro_stat:.6f}")
    print(f"  p-value = {shapiro_p:.6f}")
    
    if shapiro_p > 0.05:
        print(f"  ✅ НЕЛЬЗЯ ОТВЕРГНУТЬ нормальность (p > 0.05)")
    else:
        print(f"  ⚠️ Распределение отклоняется от нормального")
    
    # Автокорреляция residuals
    max_lag = 20
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        corr, _ = pearsonr(residuals[:-lag], residuals[lag:])
        acf.append(corr)
    
    print(f"\nАвтокорреляция residuals:")
    for lag in [1, 2, 3, 5, 10]:
        print(f"  lag={lag:2d}: r = {acf[lag]:+.6f}")
    
    # Визуализация
    ax3 = axes[0, 2]
    ax3.hist(residuals_sample, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    x_plot = np.linspace(-3*sigma, 3*sigma, 200)
    ax3.plot(x_plot, norm.pdf(x_plot, 0, sigma), 'r-', linewidth=2, label=f'𝒩(0, {sigma:.3f}²)')
    ax3.set_xlabel('Residuals = Δm - μ(PF)')
    ax3.set_ylabel('Плотность')
    ax3.set_title(f'Распределение residuals (p={shapiro_p:.4f})')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    ax4 = axes[1, 0]
    ax4.bar(range(max_lag + 1), acf, color='steelblue', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5, label='95% ДИ')
    ax4.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Лаг')
    ax4.set_ylabel('Автокорреляция')
    ax4.set_title('ACF residuals')
    ax4.legend()
    ax4.grid(alpha=0.3)

# ============================================================================
# ЧАСТЬ 3: СИМУЛЯЦИЯ СДУ И СРАВНЕНИЕ С РЕАЛЬНЫМИ ДАННЫМИ
# ============================================================================
print("\n" + "=" * 80)
print("ЧАСТЬ 3: СИМУЛЯЦИЯ СДУ И СРАВНЕНИЕ")
print("=" * 80)

if popt is not None:
    # Параметры симуляции
    n_steps = 100000
    phi_0 = deltas[0]
    
    # Функция дрейфа от PF
    def drift(phi):
        # Вычисляем PF для данного phi (приближённо)
        # Поскольку PF зависит от t, а не от phi, делаем упрощение:
        # Используем среднюю высоту t ~ 500000
        t_avg = 500000
        pf_val = prime_field(t_avg + phi * 10)  # приближение
        return sigmoid(pf_val, *popt)
    
    # Упрощённая симуляция: используем эмпирический дрейф напрямую от PF
    print("\nСимуляция СДУ (упрощённая версия)...")
    
    simulated_dm = []
    simulated_deltas = [phi_0]
    
    # Используем реальные PF для симуляции (более честно)
    pf_sim = prime_fields[:n_steps]
    
    np.random.seed(42)
    for i in tqdm(range(n_steps), desc="Симуляция"):
        mu_val = sigmoid(pf_sim[i], *popt)
        # Добавляем гауссовский шум
        dm_sim = mu_val + sigma * np.random.randn()
        # Дискретизируем до {0, 1, 2}
        dm_disc = np.clip(np.round(dm_sim), 0, 2)
        simulated_dm.append(dm_disc)
    
    simulated_dm = np.array(simulated_dm)
    
    # Сравнение распределений Δm
    real_dm = diff_grams[:n_steps]
    
    # Критерий Колмогорова-Смирнова
    ks_stat, ks_p = ks_2samp(real_dm, simulated_dm)
    
    print(f"\nКритерий Колмогорова-Смирнова:")
    print(f"  KS-статистика = {ks_stat:.6f}")
    print(f"  p-value = {ks_p:.6f}")
    
    if ks_p > 0.05:
        print(f"  ✅ СДУ АДЕКВАТНО ОПИСЫВАЕТ ДАННЫЕ (p > 0.05)!")
    else:
        print(f"  ⚠️ Распределения статистически значимо различаются")
    
    # Сравнение частот
    real_freq = np.bincount(real_dm.astype(int), minlength=3)[:3] / len(real_dm)
    sim_freq = np.bincount(simulated_dm.astype(int), minlength=3)[:3] / len(simulated_dm)
    
    print(f"\nЧастоты Δm:")
    print(f"  Значение | Реальные | Симуляция | Разница")
    print(f"  ---------|----------|-----------|--------")
    for i in range(3):
        diff_pct = 100 * (sim_freq[i] - real_freq[i])
        print(f"  Δm = {i}   | {real_freq[i]:7.4f} | {sim_freq[i]:8.4f} | {diff_pct:+7.2f}%")
    
    # Визуализация
    ax5 = axes[1, 1]
    x_pos = np.arange(3)
    width = 0.35
    ax5.bar(x_pos - width/2, real_freq, width, label='Реальные', color='steelblue', alpha=0.7)
    ax5.bar(x_pos + width/2, sim_freq, width, label='Симуляция', color='orange', alpha=0.7)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['0', '1', '2'])
    ax5.set_xlabel('Δm')
    ax5.set_ylabel('Частота')
    ax5.set_title(f'Сравнение распределений (KS p={ks_p:.4f})')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')

# ============================================================================
# СВОДКА РЕЗУЛЬТАТОВ
# ============================================================================
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                ВЕРИФИКАЦИЯ НЕПРЕРЫВНОГО ПРЕДЕЛА                  ║
╠══════════════════════════════════════════════════════════════════╣
║ ЧАСТЬ 1: ЭМПИРИЧЕСКИЙ ДРЕЙФ μ(PF)                                ║
"""

if popt is not None and r2 is not None:
    summary_text += f"""
║   R² = {r2:.6f}                                                  ║
║   μ(PF) = {a:.3f}/(1+exp(-{b:.3f}(PF-{c:.3f})))+{d:.3f}         ║
"""
else:
    summary_text += """
║   ⚠️ Подгонка не удалась                                          ║
"""

summary_text += f"""
╠══════════════════════════════════════════════════════════════════╣
║ ЧАСТЬ 2: ВОЛАТИЛЬНОСТЬ σ                                         ║
"""
if popt is not None:
    summary_text += f"""
║   σ = {sigma:.6f}                                                ║
║   Тест Шапиро-Уилка: p = {shapiro_p:.6f}                         ║
"""
else:
    summary_text += """
║   ⚠️ Не оценено                                                   ║
"""

summary_text += f"""
╠══════════════════════════════════════════════════════════════════╣
║ ЧАСТЬ 3: СИМУЛЯЦИЯ СДУ                                           ║
"""
if popt is not None:
    summary_text += f"""
║   KS-тест: p = {ks_p:.6f}                                        ║
"""
else:
    summary_text += """
║   ⚠️ Не выполнено                                                 ║
"""

summary_text += """
╠══════════════════════════════════════════════════════════════════╣
║ ВЫВОД:                                                           ║
"""
if popt is not None and r2 is not None and r2 > 0.7 and ks_p > 0.05:
    summary_text += """
║ ✅ НЕПРЕРЫВНЫЙ ПРЕДЕЛ ПОДТВЕРЖДЁН!                               ║
║    Грамм-диффузия адекватно описывается СДУ.                     ║
"""
elif popt is not None and r2 is not None and r2 > 0.7:
    summary_text += """
║ ⚠️ ЧАСТИЧНОЕ ПОДТВЕРЖДЕНИЕ                                       ║
║    Дрейф восстановлен, но симуляция требует уточнения.           ║
"""
else:
    summary_text += """
║ ⚠️ ТРЕБУЕТСЯ ДОРАБОТКА                                           ║
║    Необходимо больше данных или другая параметризация.           ║
"""

summary_text += """
╚══════════════════════════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fokker_planck_verification.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранён как 'fokker_planck_verification.png'")

plt.show()

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)

"""
===============================================================================
ЧАСТЬ 4: СТАЦИОНАРНОЕ РЕШЕНИЕ УРАВНЕНИЯ ФОККЕРА-ПЛАНКА
===============================================================================
Шаг 1: Построение μ(φ) через усреднение PF по бинам φ
Шаг 2: Аналитическое решение стационарного уравнения
Шаг 3: Сравнение с эмпирическим распределением Δ
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ЗАГРУЗКА ДАННЫХ (предполагаем, что deltas и prime_fields уже вычислены)
# ============================================================================
# ВАЖНО: этот код должен выполняться после Части 1-3, где уже есть:
# deltas, prime_fields, sigma, popt (параметры сигмоиды)

print("=" * 80)
print("ЧАСТЬ 4: СТАЦИОНАРНОЕ РЕШЕНИЕ УРАВНЕНИЯ ФОККЕРА-ПЛАНКА")
print("=" * 80)

# ============================================================================
# ШАГ 1: ПОСТРОЕНИЕ μ(φ) ЧЕРЕЗ УСРЕДНЕНИЕ PF ПО БИНАМ φ
# ============================================================================
print("\n" + "=" * 80)
print("ШАГ 1: ПОСТРОЕНИЕ μ(φ)")
print("=" * 80)

# Разбиваем φ = Δ на бины
n_bins_phi = 40
phi_bins = np.linspace(-0.5, 0.5, n_bins_phi + 1)
phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2

# Для каждого бина φ вычисляем средний PF
pf_by_phi = []
pf_std_by_phi = []
counts_by_phi = []

for i in range(n_bins_phi):
    mask = (deltas >= phi_bins[i]) & (deltas < phi_bins[i+1])
    n = np.sum(mask)
    counts_by_phi.append(n)
    
    if n > 100:
        pf_mean = np.mean(prime_fields[mask])
        pf_std = np.std(prime_fields[mask])
        pf_by_phi.append(pf_mean)
        pf_std_by_phi.append(pf_std)
    else:
        pf_by_phi.append(np.nan)
        pf_std_by_phi.append(np.nan)

pf_by_phi = np.array(pf_by_phi)
counts_by_phi = np.array(counts_by_phi)

# Интерполируем PF(φ) для всех φ
valid = ~np.isnan(pf_by_phi)
phi_valid = phi_centers[valid]
pf_valid = pf_by_phi[valid]

# Подгонка: PF(φ) = A * sin(B * φ + C) + D (эвристика)
def pf_of_phi(phi, A, B, C, D):
    return A * np.sin(B * phi + C) + D

try:
    popt_pf, _ = curve_fit(pf_of_phi, phi_valid, pf_valid, 
                           p0=[3.0, 10.0, 0.0, 0.0], maxfev=5000)
    A, B, C, D = popt_pf
    
    pf_smooth = pf_of_phi(phi_centers, *popt_pf)
    r2_pf = 1 - np.sum((pf_valid - pf_of_phi(phi_valid, *popt_pf))**2) / np.sum((pf_valid - np.mean(pf_valid))**2)
    
    print(f"\nPF(φ) = {A:.3f} * sin({B:.3f}·φ + {C:.3f}) + {D:.3f}")
    print(f"R² = {r2_pf:.4f}")
except:
    print("\n⚠️ Подгонка PF(φ) не удалась, используем линейную интерполяцию")
    popt_pf = None
    r2_pf = None

# Функция μ(φ) через сигмоиду от PF(φ)
def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-b * (x - c))) + d

# Параметры сигмоиды из Части 1
a, b, c, d = popt  # popt из Части 1

def mu_of_phi(phi):
    """Дрейф как функция φ"""
    if popt_pf is not None:
        pf_val = pf_of_phi(phi, *popt_pf)
    else:
        # Линейная интерполяция
        pf_val = np.interp(phi, phi_valid, pf_valid)
    return sigmoid(pf_val, a, b, c, d)

# Вычисляем μ(φ) на сетке
mu_values = mu_of_phi(phi_centers)

print(f"\nμ(φ) на сетке:")
print(f"  φ ∈ [{phi_centers[0]:.3f}, {phi_centers[-1]:.3f}]")
print(f"  μ ∈ [{mu_values.min():.3f}, {mu_values.max():.3f}]")

# ============================================================================
# ШАГ 2: АНАЛИТИЧЕСКОЕ РЕШЕНИЕ СТАЦИОНАРНОГО УРАВНЕНИЯ
# ============================================================================
print("\n" + "=" * 80)
print("ШАГ 2: АНАЛИТИЧЕСКОЕ РЕШЕНИЕ ρ∞(φ)")
print("=" * 80)

# Численное интегрирование μ(φ)
phi_grid = np.linspace(-0.5, 0.5, 500)
mu_grid = mu_of_phi(phi_grid)

# Интеграл от μ(x) dx
integral_mu = cumulative_trapezoid(mu_grid, phi_grid, initial=0)

# Стационарная плотность (ненормированная)
sigma_val = 0.595953  # из Части 2
rho_unnorm = np.exp(2 * integral_mu / sigma_val**2)

# Нормировка
norm = np.trapezoid(rho_unnorm, phi_grid)
rho_inf = rho_unnorm / norm

print(f"\nНормировочная константа: {norm:.4f}")
print(f"Максимум плотности: φ = {phi_grid[np.argmax(rho_inf)]:.3f}")

# ============================================================================
# ШАГ 3: СРАВНЕНИЕ С ЭМПИРИЧЕСКИМ РАСПРЕДЕЛЕНИЕМ Δ
# ============================================================================
print("\n" + "=" * 80)
print("ШАГ 3: СРАВНЕНИЕ С ЭМПИРИКОЙ")
print("=" * 80)

# Эмпирическая гистограмма Δ
hist_emp, bin_edges = np.histogram(deltas, bins=40, range=(-0.5, 0.5), density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Теоретические значения в центрах бинов
rho_theory_bins = np.exp(2 * cumulative_trapezoid(mu_grid, phi_grid, initial=0)) / norm
rho_theory_at_bins = np.interp(bin_centers, phi_grid, rho_theory_bins)

# Корреляция
corr_fp = np.corrcoef(hist_emp, rho_theory_at_bins)[0, 1]

# KS-тест
# Генерируем выборку из теоретического распределения
n_samples = 100000
np.random.seed(42)
theory_samples = np.random.choice(phi_grid, n_samples, p=rho_inf/rho_inf.sum())
ks_stat_fp, ks_p_fp = stats.ks_2samp(deltas[:n_samples], theory_samples)

print(f"\nСравнение с эмпирическим распределением:")
print(f"  Корреляция Пирсона: r = {corr_fp:.6f}")
print(f"  KS-тест: p-value = {ks_p_fp:.6f}")

if ks_p_fp > 0.05:
    print(f"\n  ✅ ТРИУМФ! Теория неотличима от эмпирики (p > 0.05)!")
elif corr_fp > 0.95:
    print(f"\n  ✅ ОЧЕНЬ ХОРОШО! Корреляция > 0.95, форма воспроизведена!")
elif corr_fp > 0.9:
    print(f"\n  👍 ХОРОШО! Корреляция > 0.9")
else:
    print(f"\n  ⚠️ Требуется уточнение модели")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Решение уравнения Фоккера-Планка для Грамм-диффузии', fontsize=14)

# График 1: PF(φ)
ax1 = axes[0, 0]
ax1.errorbar(phi_centers[valid], pf_valid, yerr=pf_std_by_phi[valid], fmt='o', 
             markersize=3, alpha=0.7, capsize=2, label='Эмпирические')
if popt_pf is not None:
    ax1.plot(phi_centers, pf_smooth, 'r-', linewidth=2, label=f'Подгонка (R²={r2_pf:.4f})')
ax1.set_xlabel('φ = Δ')
ax1.set_ylabel('⟨PF⟩')
ax1.set_title('Средний Prime Field как функция φ')
ax1.legend()
ax1.grid(alpha=0.3)

# График 2: μ(φ)
ax2 = axes[0, 1]
ax2.plot(phi_centers, mu_values, 'b-', linewidth=2)
ax2.set_xlabel('φ = Δ')
ax2.set_ylabel('μ(φ)')
ax2.set_title('Дрейф как функция φ')
ax2.grid(alpha=0.3)

# График 3: Интеграл от μ
ax3 = axes[0, 2]
ax3.plot(phi_grid, integral_mu, 'g-', linewidth=2)
ax3.set_xlabel('φ')
ax3.set_ylabel('∫ μ(x) dx')
ax3.set_title('Интеграл дрейфа')
ax3.grid(alpha=0.3)

# График 4: ρ∞(φ) vs эмпирика
ax4 = axes[1, 0]
ax4.hist(deltas, bins=40, range=(-0.5, 0.5), density=True, alpha=0.5, 
         color='gray', label='Эмпирическое')
ax4.plot(phi_grid, rho_inf, 'r-', linewidth=2, label='Фоккер-Планк')
ax4.set_xlabel('φ = Δ')
ax4.set_ylabel('Плотность')
ax4.set_title(f'Стационарное распределение (r={corr_fp:.4f})')
ax4.legend()
ax4.grid(alpha=0.3)

# График 5: Q-Q plot
ax5 = axes[1, 1]
n_quantiles = 100
emp_quantiles = np.percentile(deltas[:n_samples], np.linspace(1, 99, n_quantiles))
theory_quantiles = np.percentile(theory_samples, np.linspace(1, 99, n_quantiles))
ax5.scatter(theory_quantiles, emp_quantiles, alpha=0.5, s=5, color='blue')
ax5.plot([-0.5, 0.5], [-0.5, 0.5], 'r--', linewidth=1, label='y = x')
ax5.set_xlabel('Теоретические квантили')
ax5.set_ylabel('Эмпирические квантили')
ax5.set_title(f'Q-Q plot (KS p={ks_p_fp:.4f})')
ax5.legend()
ax5.grid(alpha=0.3)

# График 6: Сводка
ax6 = axes[1, 2]
ax6.axis('off')

summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║            РЕШЕНИЕ УРАВНЕНИЯ ФОККЕРА-ПЛАНКА                      ║
╠══════════════════════════════════════════════════════════════════╣
║ ПАРАМЕТРЫ:                                                       ║
║   σ = {sigma_val:.6f}                                            ║
║   PF(φ) подгонка: R² = {r2_pf if r2_pf else 'N/A':>8}            ║
║                                                                  ║
║ СРАВНЕНИЕ С ЭМПИРИКОЙ:                                           ║
║   Корреляция Пирсона: r = {corr_fp:.6f}                          ║
║   KS-тест: p = {ks_p_fp:.6f}                                     ║
╠══════════════════════════════════════════════════════════════════╣
║ ВЫВОД:                                                           ║
"""
if ks_p_fp > 0.05:
    summary_text += """
║ ✅ ТРИУМФ! Уравнение Фоккера-Планка точно описывает данные!       ║
"""
elif corr_fp > 0.95:
    summary_text += """
║ ✅ ОТЛИЧНО! Форма распределения воспроизведена (r > 0.95)         ║
"""
elif corr_fp > 0.9:
    summary_text += """
║ 👍 ХОРОШО! Качественное согласие достигнуто                       ║
"""
else:
    summary_text += """
║ ⚠️ Требуется уточнение модели                                     ║
"""

summary_text += """
╚══════════════════════════════════════════════════════════════════╝
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fokker_planck_solution.png', dpi=150, bbox_inches='tight')
print("\n✓ График сохранён как 'fokker_planck_solution.png'")

plt.show()

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)