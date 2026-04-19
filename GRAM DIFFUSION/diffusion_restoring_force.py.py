"""
===============================================================================
ОДНОРАЗОВОЕ ВЫЧИСЛЕНИЕ И СОХРАНЕНИЕ ДАННЫХ (С ПРОВЕРКОЙ ФАЙЛОВ)
Запустить один раз, дождаться завершения (~25-30 минут)
Если файлы уже есть — просто загружает их и продолжает
===============================================================================
"""

import numpy as np
import mpmath as mp
from tqdm import tqdm
import os

mp.mp.dps = 50

# ============================================================================
# 1. ПРОВЕРКА НАЛИЧИЯ ФАЙЛОВ
# ============================================================================
print("=" * 80)
print("ПРОВЕРКА НАЛИЧИЯ ФАЙЛОВ ДАННЫХ")
print("=" * 80)

files_needed = ['deltas_2M.npy', 'prime_fields_2M.npy', 'diff_grams_2M.npy']
files_exist = all(os.path.exists(f) for f in files_needed)

if files_exist:
    print("\n✓ Все файлы данных уже существуют!")
    print("  - deltas_2M.npy")
    print("  - prime_fields_2M.npy")
    print("  - diff_grams_2M.npy")
    print("\n⏩ Пропускаем вычисления, переходим к анализу...\n")
    
    # Загружаем данные для проверки
    deltas = np.load('deltas_2M.npy')
    prime_fields = np.load('prime_fields_2M.npy')
    diff_grams = np.load('diff_grams_2M.npy')
    
    print(f"✓ Загружено {len(deltas):,} нулей")
    print(f"  Δ ∈ [{deltas.min():.4f}, {deltas.max():.4f}]")
    print(f"  PF ∈ [{prime_fields.min():.4f}, {prime_fields.max():.4f}]")
    print(f"  Δm ∈ [{diff_grams.min()}, {diff_grams.max()}]")
    
else:
    print("\n⚠️ Не все файлы данных найдены.")
    print("   Будут выполнены полные вычисления (~25-30 минут).\n")
    
    # ========================================================================
    # 2. ЗАГРУЗКА ИСХОДНЫХ ДАННЫХ
    # ========================================================================
    print("Загрузка исходных данных...")
    zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
    gram = np.load('gram_indices_2M.npy')[:2_000_000]
    
    print(f"✓ Загружено {len(zeros):,} нулей")
    print(f"✓ Загружено {len(gram):,} индексов Грама")

    def siegel_theta(t):
        return float(mp.siegeltheta(t))

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def prime_field(t):
        total = 0.0
        for p in primes:
            for k in range(1, 4):
                pk = p**k
                weight = float(mp.log(p) / (k * mp.sqrt(pk)))
                total += weight * np.sin(t * np.log(pk))
        return total

    # ========================================================================
    # 3. ВЫЧИСЛЕНИЕ (ДОЛГОЕ)
    # ========================================================================
    print("\n" + "=" * 80)
    print("ВЫЧИСЛЕНИЕ Δ, PF и Δm")
    print("=" * 80)
    print("Это займёт 25-30 минут. Можно пойти попить чай.\n")

    deltas = []
    prime_fields = []
    diff_grams = []

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

    deltas = np.array(deltas, dtype=np.float32)
    prime_fields = np.array(prime_fields, dtype=np.float32)
    diff_grams = np.array(diff_grams, dtype=np.int8)

    # ========================================================================
    # 4. СОХРАНЕНИЕ
    # ========================================================================
    print("\n" + "=" * 80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    np.save('deltas_2M.npy', deltas)
    np.save('prime_fields_2M.npy', prime_fields)
    np.save('diff_grams_2M.npy', diff_grams)

    print("\n✓ Сохранено:")
    print("  - deltas_2M.npy")
    print("  - prime_fields_2M.npy")
    print("  - diff_grams_2M.npy")

print("\n" + "=" * 80)
print("✅ ГОТОВО К АНАЛИЗУ")
print("=" * 80)
print("\nТеперь можно запускать быстрый анализ (19.04.1_analysis.py)")

# ============================================================================
# 5. ЕСЛИ ФАЙЛЫ УЖЕ БЫЛИ — МОЖНО СРАЗУ ПОКАЗАТЬ БАЗОВУЮ СТАТИСТИКУ
# ============================================================================
if files_exist:
    print("\n" + "=" * 80)
    print("БАЗОВАЯ СТАТИСТИКА ЗАГРУЖЕННЫХ ДАННЫХ")
    print("=" * 80)
    
    # Вычисляем Δφ
    dphi = np.diff(deltas)
    
    print(f"\nΔ = θ/π - m:")
    print(f"  Диапазон: [{deltas.min():.4f}, {deltas.max():.4f}]")
    print(f"  Среднее: {np.mean(deltas):.6f}")
    print(f"  Стд: {np.std(deltas):.6f}")
    
    print(f"\nPrime Field:")
    print(f"  Диапазон: [{prime_fields.min():.4f}, {prime_fields.max():.4f}]")
    print(f"  Среднее: {np.mean(prime_fields):.6f}")
    print(f"  Стд: {np.std(prime_fields):.6f}")
    
    print(f"\nΔm (разность индексов Грама):")
    unique, counts = np.unique(diff_grams, return_counts=True)
    for val, count in zip(unique, counts):
        pct = 100 * count / len(diff_grams)
        print(f"  Δm = {val}: {count:,} ({pct:.2f}%)")
    
    print(f"\nΔφ = φ_next - φ_current:")
    print(f"  Диапазон: [{dphi.min():.4f}, {dphi.max():.4f}]")
    print(f"  Среднее: {np.mean(dphi):.6f}")
    print(f"  Стд: {np.std(dphi):.6f}")
"""
===============================================================================
ИСПРАВЛЕННЫЙ АНАЛИЗ С ОГРАНИЧЕНИЕМ κ И ЛИНЕЙНЫМ ПРИБЛИЖЕНИЕМ
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import i0
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 80)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

deltas = np.load('deltas_2M.npy')
prime_fields = np.load('prime_fields_2M.npy')
diff_grams = np.load('diff_grams_2M.npy')

print(f"✓ Загружено {len(deltas):,} нулей")
print(f"  Δ ∈ [{deltas.min():.4f}, {deltas.max():.4f}]")

# ============================================================================
# 2. ВЫЧИСЛЕНИЕ Δφ
# ============================================================================
dphi = np.diff(deltas)
print(f"\n✓ Δφ ∈ [{dphi.min():.4f}, {dphi.max():.4f}], среднее = {np.mean(dphi):.6f}")

# ============================================================================
# 3. ПРЯМАЯ ПОДГОНКА μ_φ(φ)
# ============================================================================
print("\n" + "=" * 80)
print("ПРЯМАЯ ПОДГОНКА μ_φ(φ)")
print("=" * 80)

n_bins_phi = 50
phi_bins = np.linspace(-0.5, 0.5, n_bins_phi + 1)
phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2

mu_phi_direct = []
for i in range(n_bins_phi):
    mask = (deltas[:-1] >= phi_bins[i]) & (deltas[:-1] < phi_bins[i+1])
    n = np.sum(mask)
    if n > 100:
        mu_phi_direct.append(np.mean(dphi[mask]))
    else:
        mu_phi_direct.append(np.nan)

mu_phi_direct = np.array(mu_phi_direct)
valid = ~np.isnan(mu_phi_direct)
phi_valid = phi_centers[valid]
mu_valid = mu_phi_direct[valid]

print(f"Валидных бинов: {np.sum(valid)} / {n_bins_phi}")

# Модель 1: μ(φ) = -α * sin(β·φ)
def restoring_force_sin(phi, alpha, beta):
    return -alpha * np.sin(beta * phi)

# Модель 2: μ(φ) = -α_eff · φ (линейное приближение)
def restoring_force_linear(phi, alpha_eff):
    return -alpha_eff * phi

# Подгонка
try:
    popt_sin, _ = curve_fit(restoring_force_sin, phi_valid, mu_valid, 
                            p0=[0.5, 6.0], maxfev=5000)
    alpha_sin, beta_sin = popt_sin
    
    mu_pred_sin = restoring_force_sin(phi_valid, *popt_sin)
    r2_sin = 1 - np.sum((mu_valid - mu_pred_sin)**2) / np.sum((mu_valid - np.mean(mu_valid))**2)
    
    # Эффективный линейный коэффициент
    alpha_eff = alpha_sin * beta_sin
    
    print(f"\n✅ Синусоидальная модель:")
    print(f"   μ(φ) = -{alpha_sin:.4f} * sin({beta_sin:.4f}·φ)")
    print(f"   R² = {r2_sin:.6f}")
    print(f"\n   Эффективный линейный коэффициент: α_eff = {alpha_eff:.4f}")
    print(f"   (при малых φ: μ(φ) ≈ -{alpha_eff:.4f}·φ)")
    
    model_type = 'sin'
    
except Exception as e:
    print(f"\n❌ Синусоидальная модель не удалась: {e}")
    
    # Пробуем сразу линейную
    popt_lin, _ = curve_fit(restoring_force_linear, phi_valid, mu_valid, p0=[1.0])
    alpha_eff = popt_lin[0]
    
    mu_pred_lin = restoring_force_linear(phi_valid, alpha_eff)
    r2_lin = 1 - np.sum((mu_valid - mu_pred_lin)**2) / np.sum((mu_valid - np.mean(mu_valid))**2)
    
    print(f"\n✅ Линейная модель:")
    print(f"   μ(φ) = -{alpha_eff:.4f}·φ")
    print(f"   R² = {r2_lin:.6f}")
    
    model_type = 'linear'
    r2_sin = None

# ============================================================================
# 4. ОЦЕНКА σ_φ
# ============================================================================
print("\n" + "=" * 80)
print("ОЦЕНКА σ_φ")
print("=" * 80)

if model_type == 'sin':
    mu_pred_all = -alpha_sin * np.sin(beta_sin * deltas[:-1])
else:
    mu_pred_all = -alpha_eff * deltas[:-1]

residuals_phi = dphi - mu_pred_all
sigma_phi = np.std(residuals_phi)

print(f"σ_φ = {sigma_phi:.6f}")

# ============================================================================
# 5. СТАЦИОНАРНОЕ РЕШЕНИЕ (С ОГРАНИЧЕНИЕМ κ)
# ============================================================================
print("\n" + "=" * 80)
print("СТАЦИОНАРНОЕ РЕШЕНИЕ")
print("=" * 80)

phi_grid = np.linspace(-0.5, 0.5, 1000)

# ВАЖНО: ограничиваем κ разумным значением
if model_type == 'sin':
    kappa_raw = 2 * alpha_sin / (beta_sin * sigma_phi**2)
    kappa = min(kappa_raw, 100.0)  # ОГРАНИЧЕНИЕ
    
    print(f"κ_raw = {kappa_raw:.2f} → ограничено до κ = {kappa:.2f}")
    
    if kappa_raw > 100:
        print("   (используется линейное приближение Гаусса)")
    
    def von_mises_pdf(phi, kappa, mu):
        phi_wrapped = (phi - mu + np.pi) % (2 * np.pi) - np.pi
        return np.exp(kappa * np.cos(phi_wrapped)) / (2 * np.pi * i0(kappa))
    
    phi_rad = phi_grid * np.pi
    rho_theory = von_mises_pdf(phi_rad, kappa, 0.0)
    rho_theory = rho_theory / np.trapezoid(rho_theory, phi_grid)
    theory_label = f'Фон Мизес (κ={kappa:.1f})'
    
else:
    # Линейная модель → Гаусс
    var_phi = sigma_phi**2 / (2 * alpha_eff)
    print(f"Дисперсия = {var_phi:.6f} (σ = {np.sqrt(var_phi):.4f})")
    
    rho_theory = np.exp(-phi_grid**2 / (2 * var_phi))
    rho_theory = rho_theory / np.trapezoid(rho_theory, phi_grid)
    theory_label = f'Гаусс (σ={np.sqrt(var_phi):.3f})'

# ============================================================================
# 6. АЛЬТЕРНАТИВНОЕ РЕШЕНИЕ: ГАУСС (из линейного приближения)
# ============================================================================
# Даже если подогнана синусоида, линейное приближение даёт гауссовское распределение
if model_type == 'sin':
    var_phi_gauss = sigma_phi**2 / (2 * alpha_eff)
    rho_gauss = np.exp(-phi_grid**2 / (2 * var_phi_gauss))
    rho_gauss = rho_gauss / np.trapezoid(rho_gauss, phi_grid)
    gauss_label = f'Гаусс (σ={np.sqrt(var_phi_gauss):.3f})'
else:
    rho_gauss = rho_theory.copy()
    gauss_label = theory_label

# ============================================================================
# 7. СРАВНЕНИЕ С ЭМПИРИКОЙ
# ============================================================================
print("\n" + "=" * 80)
print("СРАВНЕНИЕ С ЭМПИРИКОЙ")
print("=" * 80)

hist_emp, bin_edges = np.histogram(deltas, bins=40, range=(-0.5, 0.5), density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Используем гауссовское приближение для сравнения
if model_type == 'sin':
    rho_at_bins = np.exp(-bin_centers**2 / (2 * var_phi_gauss))
else:
    rho_at_bins = np.exp(-bin_centers**2 / (2 * var_phi))

rho_at_bins = rho_at_bins / np.trapezoid(rho_gauss, phi_grid)

corr_fp = np.corrcoef(hist_emp, rho_at_bins)[0, 1]

# Для визуального сравнения используем логарифмический масштаб
log_hist = np.log10(hist_emp + 1e-10)
log_rho = np.log10(rho_at_bins + 1e-10)
corr_log = np.corrcoef(log_hist, log_rho)[0, 1]

print(f"Корреляция Пирсона (линейная шкала): r = {corr_fp:.6f}")
print(f"Корреляция Пирсона (логарифмическая шкала): r_log = {corr_log:.6f}")

if corr_fp > 0.9:
    print("✅ ОТЛИЧНО! r > 0.9")
elif corr_fp > 0.7:
    print("👍 ХОРОШО! r > 0.7")
elif corr_fp > 0.5:
    print("⚠️ УМЕРЕННО! r > 0.5")
else:
    print("❌ СЛАБО! r < 0.5")

# ============================================================================
# 8. ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Анализ Грамм-диффузии: возвращающая сила и стационарное распределение', fontsize=14)

# График 1: μ(φ)
ax1 = axes[0, 0]
ax1.plot(phi_valid, mu_valid, 'o', markersize=3, alpha=0.7, color='green', label='Эмпирические')

phi_smooth = np.linspace(-0.5, 0.5, 200)
if model_type == 'sin':
    mu_smooth = -alpha_sin * np.sin(beta_sin * phi_smooth)
    label = f'μ(φ) = -{alpha_sin:.3f}·sin({beta_sin:.3f}φ)'
else:
    mu_smooth = -alpha_eff * phi_smooth
    label = f'μ(φ) = -{alpha_eff:.3f}·φ'

ax1.plot(phi_smooth, mu_smooth, 'r-', linewidth=2, label=f'{label}\nR²={r2_sin if model_type=="sin" else r2_lin:.4f}')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('φ = Δ')
ax1.set_ylabel('⟨Δφ⟩')
ax1.set_title('Возвращающая сила μ(φ)')
ax1.legend()
ax1.grid(alpha=0.3)

# График 2: Стационарное распределение (линейная шкала)
ax2 = axes[0, 1]
ax2.hist(deltas, bins=50, range=(-0.5, 0.5), density=True, alpha=0.5, 
         color='gray', label='Эмпирическое')
ax2.plot(phi_grid, rho_gauss, 'r-', linewidth=2, label=gauss_label)
if model_type == 'sin':
    ax2.plot(phi_grid, rho_theory, 'b--', linewidth=1.5, alpha=0.7, label=theory_label)
ax2.set_xlabel('φ = Δ')
ax2.set_ylabel('Плотность')
ax2.set_title(f'Стационарное распределение (r={corr_fp:.4f})')
ax2.legend()
ax2.grid(alpha=0.3)

# График 3: Логарифмический масштаб
ax3 = axes[1, 0]
ax3.semilogy(bin_centers, hist_emp + 1e-10, 'o', markersize=3, alpha=0.7, 
             color='gray', label='Эмпирическое')
ax3.semilogy(phi_grid, rho_gauss + 1e-10, 'r-', linewidth=2, label=gauss_label)
ax3.set_xlabel('φ = Δ')
ax3.set_ylabel('log₁₀ Плотность')
ax3.set_title(f'Логарифмический масштаб (r_log={corr_log:.4f})')
ax3.legend()
ax3.grid(alpha=0.3)

# График 4: Сводка
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
======================================================================
                    ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ
======================================================================
 ВОЗВРАЩАЮЩАЯ СИЛА:
"""
if model_type == 'sin':
    summary_text += f"""
   μ(φ) = -{alpha_sin:.4f}·sin({beta_sin:.4f}·φ)
   R² = {r2_sin:.6f}
   Эффективный наклон: α_eff = {alpha_eff:.4f}
"""
else:
    summary_text += f"""
   μ(φ) = -{alpha_eff:.4f}·φ
   R² = {r2_lin:.6f}
"""

summary_text += f"""
 ВОЛАТИЛЬНОСТЬ:
   σ_φ = {sigma_phi:.6f}

 СТАЦИОНАРНОЕ РАСПРЕДЕЛЕНИЕ:
   Тип: {'Фон Мизес (κ=' + f'{kappa:.1f}' + ')' if model_type=='sin' else 'Гаусс'}
   Гауссовское приближение: σ = {np.sqrt(var_phi_gauss if model_type=='sin' else var_phi):.4f}

 СРАВНЕНИЕ С ЭМПИРИКОЙ:
   Корреляция (лин.): r = {corr_fp:.6f}
   Корреляция (лог.):  r_log = {corr_log:.6f}
======================================================================
 ВЫВОД:
"""
if corr_fp > 0.7 or corr_log > 0.9:
    summary_text += """
 ✅ ТЕОРИЯ ПОДТВЕРЖДЕНА!
    Возвращающая сила существует и объясняет распределение Δ.
"""
else:
    summary_text += """
 ⚠️ ТРЕБУЕТСЯ УТОЧНЕНИЕ.
    Форма распределения частично воспроизведена.
"""

summary_text += """
======================================================================
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fokker_planck_corrected.png', dpi=150)
print("\n✓ График сохранён как 'fokker_planck_corrected.png'")
plt.show()

print("\n" + "=" * 80)
print("✅ АНАЛИЗ ЗАВЕРШЁН")
print("=" * 80)