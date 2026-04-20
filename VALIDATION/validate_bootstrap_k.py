"""
===============================================================================
РЕШАЮЩИЙ ЭКСПЕРИМЕНТ: ПРОВЕРКА БУТСТРЭПА НА zeros_2M.txt
===============================================================================
"""
import numpy as np
from scipy.stats import gamma
import mpmath as mp

mp.mp.dps = 50

def siegel_theta(t):
    return float(mp.siegeltheta(t))

def get_gram_index(t):
    return int(round(siegel_theta(t) / np.pi))

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("=" * 80)
print("ЗАГРУЗКА zeros_2M.txt")
print("=" * 80)

zeros = np.loadtxt('zeros_2M.txt')[:2_000_000]
print(f"✓ Загружено {len(zeros):,} нулей")

try:
    gram = np.load('gram_indices_2M.npy')
    print(f"✓ Загружены индексы Грама из кэша")
except:
    print("Вычисление индексов Грама...")
    gram = np.array([get_gram_index(t) for t in zeros])
    np.save('gram_indices_2M.npy', gram)

classes = gram % 12

# ============================================================================
# ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ k (MLE)
# ============================================================================
def compute_k_mle(indices, classes_arr, zeros_arr):
    """Вычисляет k через MLE для заданных индексов."""
    ks = []
    for c in range(12):
        mask = classes_arr[indices] == c
        if np.sum(mask) > 5:
            class_indices = indices[mask]
            unique_idx = np.unique(class_indices)
            
            if len(unique_idx) > 5:
                sorted_idx = np.sort(unique_idx)
                class_zeros = zeros_arr[sorted_idx]
                intervals = np.diff(class_zeros)
                intervals = intervals[intervals > 0]
                
                if len(intervals) > 3:
                    mean_int = np.mean(intervals)
                    if mean_int > 0:
                        norm_int = intervals / mean_int
                        norm_int = norm_int[norm_int < 10]
                        
                        if len(norm_int) > 3:
                            try:
                                shape, _, _ = gamma.fit(norm_int, floc=0)
                                if 0.3 < shape < 5.0:
                                    ks.append(shape)
                            except:
                                pass
    return np.mean(ks) if ks else np.nan

# ============================================================================
# ЭКСПЕРИМЕНТ: ВЫБОРКИ РАЗНОГО РАЗМЕРА
# ============================================================================
print("\n" + "=" * 80)
print("ЭКСПЕРИМЕНТ: ЗАВИСИМОСТЬ k ОТ РАЗМЕРА ВЫБОРКИ")
print("=" * 80)

sample_sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
results = []

for sample_size in sample_sizes:
    print(f"\n--- Размер выборки: {sample_size} ---")
    
    # Берём 10 разных подвыборок для стабильности
    ks_direct = []
    ks_bootstrap = []
    
    for seed in range(10):
        np.random.seed(seed)
        test_indices = np.random.choice(len(zeros), sample_size, replace=False)
        
        # Прямое вычисление
        k_direct = compute_k_mle(test_indices, classes, zeros)
        if not np.isnan(k_direct):
            ks_direct.append(k_direct)
        
        # Бутстрэп с уникализацией
        boot_vals = []
        for _ in range(50):
            boot_idx = np.random.choice(test_indices, sample_size, replace=True)
            unique_idx = np.unique(boot_idx)
            k_val = compute_k_mle(unique_idx, classes, zeros)
            if not np.isnan(k_val):
                boot_vals.append(k_val)
        
        if boot_vals:
            ks_bootstrap.append(np.mean(boot_vals))
    
    if ks_direct and ks_bootstrap:
        mean_direct = np.mean(ks_direct)
        std_direct = np.std(ks_direct)
        mean_boot = np.mean(ks_bootstrap)
        std_boot = np.std(ks_bootstrap)
        
        print(f"  Прямое:    {mean_direct:.4f} ± {std_direct:.4f}")
        print(f"  Бутстрэп:  {mean_boot:.4f} ± {std_boot:.4f}")
        print(f"  Разница:   {mean_direct - mean_boot:+.4f}")
        
        results.append({
            'size': sample_size,
            'direct_mean': mean_direct,
            'direct_std': std_direct,
            'boot_mean': mean_boot,
            'boot_std': std_boot
        })
    else:
        print(f"  ❌ Не удалось вычислить")

# ============================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 80)
print("СВОДКА РЕЗУЛЬТАТОВ")
print("=" * 80)

print(f"\n{'Размер':<10} {'Прямое k':<15} {'Бутстрэп k':<15} {'Разница':<12} {'Статус':<10}")
print("-" * 65)

for r in results:
    diff = r['direct_mean'] - r['boot_mean']
    status = "✓ ОК" if abs(diff) < 0.1 else "⚠️ РАСХОЖДЕНИЕ"
    print(f"{r['size']:<10} {r['direct_mean']:<15.4f} {r['boot_mean']:<15.4f} {diff:+12.4f} {status:<10}")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================
import matplotlib.pyplot as plt

sizes = [r['size'] for r in results]
direct_means = [r['direct_mean'] for r in results]
direct_stds = [r['direct_std'] for r in results]
boot_means = [r['boot_mean'] for r in results]
boot_stds = [r['boot_std'] for r in results]

plt.figure(figsize=(10, 6))
plt.errorbar(sizes, direct_means, yerr=direct_stds, fmt='bo-', label='Прямое', capsize=5)
plt.errorbar(sizes, boot_means, yerr=boot_stds, fmt='rs-', label='Бутстрэп (с unique)', capsize=5)
plt.axhline(y=1.0, color='gray', linestyle=':', label='k=1 (Пуассон)')
plt.axhline(y=1.2, color='purple', linestyle='--', label='k=1.2 (12/10)')
plt.xlabel('Размер выборки')
plt.ylabel('Параметр k')
plt.title('Зависимость оценки k от размера выборки')
plt.xscale('log')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('k_vs_sample_size.png', dpi=150)
print("\n✓ График сохранён как 'k_vs_sample_size.png'")

# ============================================================================
# ДИАГНОСТИКА ДАННЫХ ОДЛЫЖКО
# ============================================================================
print("\n" + "=" * 80)
print("ДИАГНОСТИКА: ПОЧЕМУ НА ДАННЫХ ОДЛЫЖКО k = 0.65?")
print("=" * 80)

# Проверяем, есть ли систематическое различие между данными
print("\nВозможные причины k = 0.65 на данных Одлыжко:")
print("  1. Слишком маленькая выборка (10 000 нулей)")
print("  2. Неправильная целая часть BASE")
print("  3. Дубликаты при бутстрэпе (уже исправлено unique)")
print("  4. Реальная аномалия на этой высоте")

# Смотрим на распределение интервалов
print("\nДля проверки нужны данные Одлыжко большего объёма (100 000+ нулей)")

plt.show()