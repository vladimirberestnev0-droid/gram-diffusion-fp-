from mpmath import dirichlet, findroot, mp, im, loggamma, log, pi as mp_pi
from scipy import stats
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import time

mp.dps = 15

def theta_L_mod3(t):
    gamma_term = im(loggamma(0.75 + 0.5j * t))
    log_term = -(t/2) * log(mp_pi/3)
    return float(gamma_term + log_term)

def gram_index(t):
    return int(round(float(theta_L_mod3(t)) / float(mp_pi)))

def L_chi3_real(t):
    s = 0.5 + 1j * t
    try:
        # Явное задание характера Дирихле mod 3: χ(1)=1, χ(2)=-1, χ(0)=0
        return float(dirichlet(s, [1, -1, 0]).real)
    except:
        try:
            return float(dirichlet(s, 1, 3).real)
        except:
            return float(dirichlet(1, 3, s).real)

def find_zero_in_interval(start_t):
    """Находит один ноль начиная с start_t"""
    try:
        z = findroot(L_chi3_real, start_t, tol=1e-8, maxsteps=50)
        return float(z)
    except:
        return None

def compute_zeros_adaptive(count, t_start=6.0, batch_size=1000):
    """Адаптивный поиск с батчами"""
    zeros = []
    t_current = t_start
    n_cores = cpu_count()
    
    print(f"Запуск на {n_cores} ядрах")
    print(f"Цель: {count} нулей L-функции mod 3\n")
    
    while len(zeros) < count:
        # Оцениваем нужный интервал
        remaining = count - len(zeros)
        # Плотность нулей для L-mod-3: ~0.5/π * log(t*π/3)
        gram_per_unit = 0.5 / float(mp_pi) * np.log(4 * float(mp_pi) / 3)
        interval_length = max(50, remaining / gram_per_unit * 1.2)
        
        t_end = t_current + interval_length
        
        print(f"  Поиск в [{t_current:.1f}, {t_end:.1f}]... ", end='', flush=True)
        
        # Параллельный поиск в интервале
        step = 0.3  # Шаг для начальных точек
        start_points = np.arange(t_current, t_end, step)
        
        with Pool(n_cores) as pool:
            batch_zeros = pool.map(find_zero_in_interval, start_points)
        
        # Обработка результатов
        batch_zeros = sorted(set([z for z in batch_zeros if z is not None and z > t_current]))
        filtered = []
        for z in batch_zeros:
            if not filtered or z - filtered[-1] > 0.01:
                filtered.append(z)
        
        zeros.extend(filtered)
        zeros = sorted(set(zeros))  # Финальная дедупликация
        
        print(f"найдено {len(filtered)} (всего {len(zeros)})")
        
        # Сохраняем каждые 2000 нулей
        if len(zeros) % 2000 == 0 or len(zeros) >= count:
            np.save(f'zeros_Lmod3_{len(zeros)}.npy', zeros[:count])
            print(f"  ✓ Сохранено {min(len(zeros), count)} нулей")
        
        if filtered:
            t_current = filtered[-1] + 0.5
        else:
            t_current = t_end
    
    return np.array(zeros[:count])

def find_optimal_modulus(zeros, max_mod=24):
    """Находит модуль N, дающий максимальную структуру"""
    gram_indices = np.array([gram_index(z) for z in zeros])
    
    results = []
    for N in range(2, max_mod + 1):
        classes = gram_indices % N
        counts = np.bincount(classes, minlength=N)
        
        # Хи-квадрат тест
        expected = len(zeros) / N
        chi2 = np.sum((counts - expected)**2 / expected)
        p_value = 1 - stats.chi2.cdf(chi2, df=N-1)
        
        # Средняя дисперсия внутри классов
        variances = []
        for c in range(N):
            mask = (classes == c)
            class_zeros = zeros[mask]
            if len(class_zeros) >= 5:
                intervals = np.diff(class_zeros)
                s_norm = intervals / np.mean(intervals)
                variances.append(np.var(s_norm, ddof=1))
        
        mean_var = np.mean(variances) if variances else np.inf
        mean_k = 1.0 / mean_var if mean_var > 0 else 0
        
        results.append({
            'N': N,
            'p_value': p_value,
            'chi2': chi2,
            'mean_var': mean_var,
            'mean_k': mean_k
        })
        
        print(f"N={N:2d}: p={p_value:.4f}, χ²={chi2:.2f}, Var={mean_var:.4f}, k={mean_k:.4f}")
    
    return results

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    print("="*60)
    print("ПРОВЕРКА ГИПОТЕЗЫ k_inf = N/10 ДЛЯ L-mod-3")
    print(f"Ядер процессора: {cpu_count()}")
    print("="*60)
    
    start_time = time.time()
    
    # Генерация 11400 нулей
    n_zeros = 11400
    zeros = compute_zeros_adaptive(n_zeros, t_start=6.0, batch_size=2000)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Генерация {n_zeros} нулей завершена за {elapsed/60:.1f} минут")
    print(f"  Среднее время на ноль: {1000*elapsed/n_zeros:.2f} мс")
    
    # Сохраняем финальный результат
    np.save('zeros_Lmod3_final.npy', zeros)
    
    print("\n" + "="*60)
    print("ПОИСК ОПТИМАЛЬНОГО МОДУЛЯ N")
    print("="*60)
    
    results = find_optimal_modulus(zeros, max_mod=24)
    
    # Находим N с минимальным p-value
    best = min(results, key=lambda x: x['p_value'])
    print("\n" + "="*60)
    print(f"ОПТИМАЛЬНЫЙ МОДУЛЬ: N = {best['N']}")
    print(f"  p-value = {best['p_value']:.6f}")
    print(f"  k = {best['mean_k']:.4f}")
    print("="*60)
    
    # Проверка гипотезы
    N_opt = best['N']
    k_obs = best['mean_k']
    k_pred = N_opt / 10
    
    print(f"\nГИПОТЕЗА k_inf = N/10:")
    print(f"  N = {N_opt}")
    print(f"  Предсказание: k = {N_opt}/10 = {k_pred:.4f}")
    print(f"  Наблюдение:   k = {k_obs:.4f}")
    print(f"  Ошибка: {abs(k_obs - k_pred):.4f} ({100*abs(k_obs - k_pred)/k_pred:.1f}%)")
    
    if abs(k_obs - k_pred) < 0.1 * k_pred:
        print("\n✓✓✓ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ!")
    elif abs(k_obs - 1.20) < 0.1:
        print(f"\n○ k ≈ 1.20 — универсальный модуль 12, а не N={N_opt}")
    else:
        print(f"\n○ Гипотеза не подтверждается на {n_zeros} нулях")