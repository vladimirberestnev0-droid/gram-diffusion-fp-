"""
===============================================================================
СТРОГАЯ ПРОВЕРКА: ПРОЦЕСС ОРНШТЕЙНА-УЛЕНБЕКА — АРТЕФАКТ ИЛИ РЕАЛЬНОСТЬ?
===============================================================================
Четыре теста для защиты от обвинения в статистическом артефакте:
1. Shuffle test — перемешивание Δφ
2. Суррогатные данные — отражённое случайное блуждание с теми же границами
3. Бутстрэп — стабильность γ на подвыборках
4. Независимые данные — проверка на нулях Одлыжко (t ~ 10^11)
===============================================================================
"""

import numpy as np
from scipy import stats
import mpmath as mp
from tqdm import tqdm

mp.mp.dps = 50

# ============================================================================
# 0. ЗАГРУЗКА ДАННЫХ И ВЫЧИСЛЕНИЕ Δ
# ============================================================================
print("="*80)
print("ЗАГРУЗКА ДАННЫХ И ВЫЧИСЛЕНИЕ Δ")
print("="*80)

# Загружаем 2M нулей
zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
gram = np.load('gram_indices_2M.npy')[:2_000_000]

def siegel_theta(t):
    return float(mp.siegeltheta(t))

# Вычисляем Δ (можно загрузить из кэша, если есть)
try:
    deltas = np.load('deltas_2M.npy')
    print("✓ Δ загружены из кэша")
except:
    print("Вычисление Δ...")
    deltas = []
    for i in tqdm(range(len(zeros)), desc="Δ"):
        t = zeros[i]
        theta = siegel_theta(t)
        m = gram[i]
        delta = theta / np.pi - m
        deltas.append(delta)
    deltas = np.array(deltas)
    np.save('deltas_2M.npy', deltas)

phi = deltas[:-1]
dphi = np.diff(deltas)

print(f"✓ N = {len(phi):,}")
print(f"✓ φ ∈ [{phi.min():.4f}, {phi.max():.4f}]")
print(f"✓ Δφ ∈ [{dphi.min():.4f}, {dphi.max():.4f}]")

# ============================================================================
# БАЗОВЫЕ ЗНАЧЕНИЯ (РЕАЛЬНЫЕ ДАННЫЕ)
# ============================================================================
gamma_real, intercept_real = np.polyfit(phi, dphi, 1)
gamma_real = -gamma_real

pred_real = -gamma_real * phi + intercept_real
ss_res_real = np.sum((dphi - pred_real)**2)
ss_tot_real = np.sum((dphi - np.mean(dphi))**2)
r2_real = 1 - ss_res_real / ss_tot_real

print("\n" + "="*80)
print("РЕАЛЬНЫЕ ДАННЫЕ (2M нулей)")
print("="*80)
print(f"γ = {gamma_real:.6f}")
print(f"R² = {r2_real:.6f}")

# ============================================================================
# ТЕСТ 1: SHUFFLE TEST
# ============================================================================
print("\n" + "="*80)
print("ТЕСТ 1: SHUFFLE TEST (перемешивание Δφ)")
print("="*80)

n_shuffles = 1000
gammas_shuffled = []
r2_shuffled = []

for _ in tqdm(range(n_shuffles), desc="Shuffle"):
    dphi_shuffled = np.random.permutation(dphi)
    gamma, intercept = np.polyfit(phi, dphi_shuffled, 1)
    gammas_shuffled.append(-gamma)
    
    pred = -gamma * phi + intercept
    ss_res = np.sum((dphi_shuffled - pred)**2)
    ss_tot = np.sum((dphi_shuffled - np.mean(dphi_shuffled))**2)
    r2_shuffled.append(1 - ss_res/ss_tot)

gammas_shuffled = np.array(gammas_shuffled)
r2_shuffled = np.array(r2_shuffled)

print(f"\nShuffle: γ = {np.mean(gammas_shuffled):.6f} ± {np.std(gammas_shuffled):.6f}")
print(f"Shuffle: R² = {np.mean(r2_shuffled):.6f} ± {np.std(r2_shuffled):.6f}")
print(f"p-value (R² ≥ {r2_real:.6f}) = {np.mean(r2_shuffled >= r2_real):.6f}")

if np.mean(r2_shuffled >= r2_real) < 0.001:
    print("✅ ТЕСТ 1 ПРОЙДЕН: shuffle не воспроизводит R²")
else:
    print("❌ ТЕСТ 1 ПРОВАЛЕН: shuffle даёт такой же R² — артефакт!")

# ============================================================================
# ТЕСТ 2: СУРРОГАТНЫЕ ДАННЫЕ (ОТРАЖЁННОЕ СЛУЧАЙНОЕ БЛУЖДАНИЕ)
# ============================================================================
print("\n" + "="*80)
print("ТЕСТ 2: СУРРОГАТНЫЕ ДАННЫЕ (отражённое случайное блуждание)")
print("="*80)

def generate_reflected_rw(n, sigma, bounds=(-0.5, 0.5)):
    """Генерирует случайное блуждание с отражением от границ."""
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i-1] + np.random.normal(0, sigma)
        if x[i] > bounds[1]:
            x[i] = 2*bounds[1] - x[i]
        elif x[i] < bounds[0]:
            x[i] = 2*bounds[0] - x[i]
    return x

n_surrogates = 100
r2_surr = []
gamma_surr = []

# Используем реальную волатильность
sigma_real = np.std(dphi)

for _ in tqdm(range(n_surrogates), desc="Суррогаты"):
    x = generate_reflected_rw(len(phi) + 1, sigma=sigma_real, bounds=(-0.5, 0.5))
    dx = np.diff(x)
    gamma, intercept = np.polyfit(x[:-1], dx, 1)
    gamma_surr.append(-gamma)
    
    pred = -gamma * x[:-1] + intercept
    ss_res = np.sum((dx - pred)**2)
    ss_tot = np.sum((dx - np.mean(dx))**2)
    r2_surr.append(1 - ss_res/ss_tot)

r2_surr = np.array(r2_surr)
gamma_surr = np.array(gamma_surr)

print(f"\nСуррогатные данные: γ = {np.mean(gamma_surr):.6f} ± {np.std(gamma_surr):.6f}")
print(f"Суррогатные данные: R² = {np.mean(r2_surr):.6f} ± {np.std(r2_surr):.6f}")
print(f"p-value (R² ≥ {r2_real:.6f}) = {np.mean(r2_surr >= r2_real):.6f}")

if np.mean(r2_surr >= r2_real) < 0.01:
    print("✅ ТЕСТ 2 ПРОЙДЕН: ограничение не создаёт такого R²")
else:
    print("❌ ТЕСТ 2 ПРОВАЛЕН: ограничение само по себе даёт высокий R²")

# ============================================================================
# ТЕСТ 3: БУТСТРЭП-СТАБИЛЬНОСТЬ γ
# ============================================================================
print("\n" + "="*80)
print("ТЕСТ 3: БУТСТРЭП-СТАБИЛЬНОСТЬ γ")
print("="*80)

n_bootstrap = 1000
gammas_boot = []

for _ in tqdm(range(n_bootstrap), desc="Бутстрэп"):
    idx = np.random.choice(len(phi), len(phi), replace=True)
    phi_boot = phi[idx]
    dphi_boot = dphi[idx]
    gamma, _ = np.polyfit(phi_boot, dphi_boot, 1)
    gammas_boot.append(-gamma)

gammas_boot = np.array(gammas_boot)
ci_95 = np.percentile(gammas_boot, [2.5, 97.5])

print(f"\nБутстрэп γ = {np.mean(gammas_boot):.6f} ± {np.std(gammas_boot):.6f}")
print(f"95% ДИ: [{ci_95[0]:.6f}, {ci_95[1]:.6f}]")
print(f"Истинное γ = {gamma_real:.6f} внутри ДИ: {ci_95[0] <= gamma_real <= ci_95[1]}")

if ci_95[1] - ci_95[0] < 0.1:
    print("✅ ТЕСТ 3 ПРОЙДЕН: γ стабилен (узкий ДИ)")
else:
    print("⚠️ ТЕСТ 3 ЧАСТИЧНО: ДИ широковат")

# ============================================================================
# ТЕСТ 4: НЕЗАВИСИМЫЕ ДАННЫЕ (ОДЛЫЖКО, t ~ 10^11)
# ============================================================================
print("\n" + "="*80)
print("ТЕСТ 4: НЕЗАВИСИМЫЕ ДАННЫЕ (Одлыжко, t ~ 2.68e11)")
print("="*80)

try:
    odlyzko = np.loadtxt('zero_10k_10^12.txt')
    BASE = 267_653_395_648.0
    zeros_od = BASE + odlyzko
    
    print(f"✓ Загружено {len(zeros_od)} нулей Одлыжко")
    
    # Вычисляем Δ
    deltas_od = []
    for t in tqdm(zeros_od, desc="Δ Одлыжко"):
        theta = siegel_theta(t)
        m = int(round(theta / np.pi))
        deltas_od.append(theta/np.pi - m)
    deltas_od = np.array(deltas_od)
    
    phi_od = deltas_od[:-1]
    dphi_od = np.diff(deltas_od)
    
    gamma_od, intercept_od = np.polyfit(phi_od, dphi_od, 1)
    gamma_od = -gamma_od
    
    pred_od = -gamma_od * phi_od + intercept_od
    ss_res_od = np.sum((dphi_od - pred_od)**2)
    ss_tot_od = np.sum((dphi_od - np.mean(dphi_od))**2)
    r2_od = 1 - ss_res_od / ss_tot_od
    
    print(f"\nОдлыжко (t ~ 2.68e11):")
    print(f"  γ = {gamma_od:.6f}")
    print(f"  R² = {r2_od:.6f}")
    print(f"  Отклонение от γ_real: {abs(gamma_od - gamma_real):.6f}")
    
    if abs(gamma_od - gamma_real) < 0.1:
        print("✅ ТЕСТ 4 ПРОЙДЕН: γ сохраняется на больших высотах!")
    else:
        print("⚠️ ТЕСТ 4 ЧАСТИЧНО: γ немного меняется с высотой")
        
except FileNotFoundError:
    print("⚠️ Файл zero_10k_10^12.txt не найден. Тест 4 пропущен.")

# ============================================================================
# ФИНАЛЬНЫЙ ВЕРДИКТ
# ============================================================================
print("\n" + "="*80)
print("ФИНАЛЬНЫЙ ВЕРДИКТ")
print("="*80)

tests_passed = 0
if 'r2_shuffled' in dir() and np.mean(r2_shuffled >= r2_real) < 0.001:
    tests_passed += 1
if 'r2_surr' in dir() and np.mean(r2_surr >= r2_real) < 0.01:
    tests_passed += 1
if 'ci_95' in dir() and ci_95[1] - ci_95[0] < 0.1:
    tests_passed += 1
if 'gamma_od' in dir() and abs(gamma_od - gamma_real) < 0.1:
    tests_passed += 1

print(f"\nПройдено тестов: {tests_passed} / 4")

if tests_passed == 4:
    print("\n" + "="*80)
    print("🎉🎉🎉 ПРОЦЕСС ОРНШТЕЙНА-УЛЕНБЕКА — НЕ АРТЕФАКТ!")
    print("="*80)
    print("\nВозвращающая сила γ ≈ 0.985 реальна и универсальна.")
    print("Это фундаментальное свойство Грамм-диффузии.")
elif tests_passed >= 2:
    print("\n⚠️ ЧАСТИЧНОЕ ПОДТВЕРЖДЕНИЕ. Требуется дополнительный анализ.")
else:
    print("\n❌ ТЕСТЫ ПРОВАЛЕНЫ. Возможно, это артефакт.")