"""
===============================================================================
ЭТАЛОННЫЙ ЧЕСТНЫЙ ТЕСТ ОПЕРАТОРА ГИЛЬБЕРТА-ПОЙИ
===============================================================================
Правила:
1. Обучающие данные (train) используются ТОЛЬКО для калибровки
2. Тестовые данные (test) НЕ используются при построении оператора
3. Предсказание делается БЕЗ перестроения оператора
4. Все параметры фиксируются на train
===============================================================================
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import mpmath as mp
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

mp.mp.dps = 50

class HonestHilbertPolyaOperator:
    """
    Абсолютно честная версия оператора Гильберта-Пойи.
    
    Принципы:
    - Оператор строится ТОЛЬКО на train_heights
    - Спектр вычисляется ТОЛЬКО на train
    - Калибровка ТОЛЬКО на train
    - Предсказание через экстраполяцию (теоретическую, не эмпирическую)
    """
    
    def __init__(self, P_PLUS1=0.584, P_PLUS2=0.416):
        self.P_PLUS1 = P_PLUS1
        self.P_PLUS2 = P_PLUS2
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Состояние после обучения
        self.is_trained = False
        self.train_heights = None
        self.train_eigvals = None
        self.calib_params = None
        self.n_train = None
        
    def _build_H12_diag(self, N):
        """Диагональная матрица из 12-потоковой структуры."""
        M = np.zeros((12, 12))
        for i in range(12):
            M[i, (i + 1) % 12] = self.P_PLUS1
            M[i, (i + 2) % 12] = self.P_PLUS2
        H12_full = (M + M.T) / 2
        eigvals = np.sort(eigh(H12_full)[0])
        return np.diag(eigvals[:N])
    
    def _prime_field(self, t, max_power=3):
        """Потенциал из простых чисел."""
        total = 0.0
        for p in self.primes:
            for k in range(1, max_power + 1):
                pk = p**k
                weight = float(mp.log(p) / (k * mp.sqrt(pk)))
                total += weight * np.sin(t * np.log(pk))
        return total
    
    def _build_operator(self, heights):
        """Строит оператор H = H12_diag + diag(prime_field)."""
        N = len(heights)
        H12_diag = self._build_H12_diag(N)
        V = np.zeros((N, N))
        for i, t in enumerate(heights):
            V[i, i] = self._prime_field(t)
        H = H12_diag + V
        return (H + H.T) / 2
    
    def fit(self, train_heights):
        """
        ОБУЧЕНИЕ: строит оператор и калибрует масштабирование.
        
        Важно: оператор строится ТОЛЬКО на train_heights.
        Спектр вычисляется ТОЛЬКО от этого оператора.
        """
        self.train_heights = np.array(train_heights)
        self.n_train = len(train_heights)
        
        # Шаг 1: строим оператор на train
        H_train = self._build_operator(train_heights)
        self.train_eigvals = np.sort(eigh(H_train)[0])
        
        # Шаг 2: калибруем масштабирующую функцию
        # γ = a * λ + b + c * (λ / log|λ|) + d * (n / log n)
        n_indices = np.arange(1, self.n_train + 1)
        
        def scale_func(lambda_vals, a, b, c, d):
            safe_lambda = np.abs(lambda_vals) + np.e
            safe_n = n_indices + np.e
            weyl_term = d * (n_indices / np.log(safe_n))
            prime_correction = c * (lambda_vals / np.log(safe_lambda))
            return a * lambda_vals + b + prime_correction + weyl_term
        
        try:
            popt, _ = curve_fit(
                scale_func,
                self.train_eigvals,
                train_heights,
                p0=[1.0, 10.0, 0.1, 5.0],
                maxfev=10000
            )
            self.calib_params = popt
            self.is_trained = True
            return popt
        except Exception as e:
            print(f"Калибровка не удалась: {e}")
            self.is_trained = False
            return None
    
    def predict_next(self, n_predict=6, method='weyl_extrapolation'):
        """
        ЧЕСТНОЕ ПРЕДСКАЗАНИЕ следующих нулей.
        
        Доступные методы:
        - 'weyl_extrapolation': использует закон Вейля + поправки от оператора
        - 'spectral_extrapolation': экстраполирует λ, затем применяет масштабирование
        - 'direct_fit': прямая экстраполяция γ(n) (контрольный метод)
        
        Важно: НИ ОДИН метод не использует test данные.
        """
        if not self.is_trained:
            raise ValueError("Сначала вызовите fit()")
        
        if method == 'weyl_extrapolation':
            return self._predict_weyl_extrapolation(n_predict)
        elif method == 'spectral_extrapolation':
            return self._predict_spectral_extrapolation(n_predict)
        elif method == 'direct_fit':
            return self._predict_direct_fit(n_predict)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def _predict_weyl_extrapolation(self, n_predict):
        """
        Метод Вейля: использует асимптотическую формулу
        γ_n ~ 2πn / log(n) с поправками от калибровки.
        
        Это САМЫЙ ЧЕСТНЫЙ метод, так как он не использует
        экстраполяцию λ (которая может быть подогнана).
        """
        a, b, c, d = self.calib_params
        
        predictions = []
        for i in range(n_predict):
            n = self.n_train + i + 1
            
            # Закон Вейля (асимптотика)
            weyl_approx = 2 * np.pi * n / np.log(n + np.e)
            
            # Оценка λ из закона Вейля (обратная функция)
            # λ ≈ (γ - b - weyl_correction) / a, где weyl_correction ~ d * n/log n
            # Упрощённо: λ_pred = (weyl_approx - b - d * n/log(n)) / a
            weyl_correction = d * n / np.log(n + np.e)
            lambda_pred = (weyl_approx - b - weyl_correction) / a
            
            # Пересчитываем γ через масштабирование (для самосогласованности)
            safe_lambda = np.abs(lambda_pred) + np.e
            safe_n = n + np.e
            prime_correction = c * (lambda_pred / np.log(safe_lambda))
            weyl_term = d * (n / np.log(safe_n))
            
            gamma_pred = a * lambda_pred + b + prime_correction + weyl_term
            
            predictions.append(gamma_pred)
        
        return np.array(predictions)
    
    def _predict_spectral_extrapolation(self, n_predict):
        """
        Спектральная экстраполяция: экстраполируем λ_n, затем применяем масштабирование.
        
        ВНИМАНИЕ: этот метод может быть нестабильным при экстраполяции.
        """
        a, b, c, d = self.calib_params
        
        # Экстраполяция λ (собственных значений)
        if self.n_train >= 2:
            # Используем последние 2 точки для линейной экстраполяции
            slope = (self.train_eigvals[-1] - self.train_eigvals[-2])
            last_lambda = self.train_eigvals[-1]
        else:
            slope = 0
            last_lambda = self.train_eigvals[-1]
        
        predictions = []
        for i in range(n_predict):
            n = self.n_train + i + 1
            lambda_pred = last_lambda + slope * (i + 1)
            
            safe_lambda = np.abs(lambda_pred) + np.e
            safe_n = n + np.e
            prime_correction = c * (lambda_pred / np.log(safe_lambda))
            weyl_term = d * (n / np.log(safe_n))
            
            gamma_pred = a * lambda_pred + b + prime_correction + weyl_term
            predictions.append(gamma_pred)
        
        return np.array(predictions)
    
    def _predict_direct_fit(self, n_predict):
        """
        Прямая экстраполяция γ(n) — контрольный метод.
        Показывает, насколько лучше работает оператор.
        """
        n_indices = np.arange(1, self.n_train + 1)
        
        # Простая степенная модель: γ = A * n^B
        def power_law(n, A, B):
            return A * (n ** B)
        
        try:
            popt, _ = curve_fit(power_law, n_indices, self.train_heights, p0=[10, 1])
            predictions = power_law(np.arange(self.n_train + 1, self.n_train + n_predict + 1), *popt)
            return predictions
        except:
            # Если не сошлось, используем линейную экстраполяцию
            slope = (self.train_heights[-1] - self.train_heights[0]) / (self.n_train - 1)
            last = self.train_heights[-1]
            predictions = [last + slope * (i + 1) for i in range(n_predict)]
            return np.array(predictions)


def run_honest_test(train_zeros=None, test_zeros=None, n_train=6, n_test=6, use_first=True):
    """
    ЭТАЛОННЫЙ ЧЕСТНЫЙ ТЕСТ ОПЕРАТОРА ГИЛЬБЕРТА-ПОЙИ
    
    Параметры:
    ----------
    train_zeros : array-like, optional
        Обучающие нули (если None, берутся из файла)
    test_zeros : array-like, optional
        Тестовые нули (если None, берутся из файла)
    n_train : int
        Количество нулей для обучения (если train_zeros=None)
    n_test : int
        Количество нулей для тестирования (если test_zeros=None)
    use_first : bool
        Если True, использует первые нули. Если False, использует случайный блок.
    """
    print("=" * 80)
    print("ЭТАЛОННЫЙ ЧЕСТНЫЙ ТЕСТ ОПЕРАТОРА ГИЛЬБЕРТА-ПОЙИ")
    print("=" * 80)
    
    # Загрузка данных (если не переданы)
    if train_zeros is None or test_zeros is None:
        try:
            zeros = np.loadtxt('zeros_2M.txt')
            print(f"\n✓ Загружено {len(zeros):,} нулей из файла")
        except:
            print("\n❌ Файл zeros_2M.txt не найден!")
            print("   Пожалуйста, укажите train_zeros и test_zeros вручную")
            return None, None, None
    
    # Определяем обучающие и тестовые нули
    if train_zeros is None:
        if use_first:
            train = zeros[:n_train]
            test = zeros[n_train:n_train + n_test] if test_zeros is None else test_zeros
        else:
            # Случайный блок
            max_start = len(zeros) - n_train - n_test
            start = np.random.randint(0, max_start)
            train = zeros[start:start + n_train]
            test = zeros[start + n_train:start + n_train + n_test]
            print(f"\n🎲 Случайный блок: нули {start+1}..{start + n_train + n_test}")
    else:
        train = np.array(train_zeros)
        test = np.array(test_zeros) if test_zeros is not None else test_zeros
        n_train = len(train)
        n_test = len(test) if test_zeros is not None else n_test
    
    print(f"\n📚 ОБУЧЕНИЕ: {n_train} нулей")
    for i, z in enumerate(train):
        print(f"  γ_{i+1} = {z:.6f}")
    
    print(f"\n🧪 ТЕСТ: {n_test} нулей (НЕ ИСПОЛЬЗУЮТСЯ при обучении)")
    if test_zeros is not None:
        for i, z in enumerate(test):
            print(f"  γ_{n_train + i + 1} = {z:.6f}")
    else:
        for i in range(n_test):
            print(f"  γ_{n_train + i + 1} = ?")
    
    # Инициализация и обучение
    hp = HonestHilbertPolyaOperator()
    params = hp.fit(train)
    
    if params is None:
        print("\n❌ Ошибка: не удалось обучить оператор")
        return None, None, None
    
    a, b, c, d = params
    print(f"\n🔧 Параметры калибровки (определены ТОЛЬКО на train):")
    print(f"   a = {a:.6f}")
    print(f"   b = {b:.6f}")
    print(f"   c = {c:.6f}")
    print(f"   d = {d:.6f}")
    
    print(f"\n📊 Спектр оператора на train:")
    for i, λ in enumerate(hp.train_eigvals[:min(6, n_train)]):
        print(f"   λ_{i+1} = {λ:.6f}")
    
    # Предсказания разными методами
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЙ")
    print("=" * 80)
    
    methods = ['spectral_extrapolation', 'direct_fit']
    method_names = ['Спектральная экстраполяция (оператор)', 'Прямая экстраполяция (контроль)']
    
    # Добавляем метод Вейля только если он работает
    try:
        weyl_pred = hp.predict_next(n_test, method='weyl_extrapolation')
        if not np.any(np.isnan(weyl_pred)) and np.mean(np.abs(weyl_pred)) < 1e6:
            methods.insert(0, 'weyl_extrapolation')
            method_names.insert(0, 'Закон Вейля + оператор')
    except:
        pass
    
    results = {}
    
    for method, name in zip(methods, method_names):
        predictions = hp.predict_next(n_test, method=method)
        
        if test_zeros is not None:
            errors = np.abs(predictions - test)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            corr, p_val = pearsonr(predictions, test)
        else:
            errors = None
            mean_error = np.nan
            max_error = np.nan
            corr = np.nan
            p_val = np.nan
        
        results[method] = {
            'predictions': predictions,
            'errors': errors,
            'mean_error': mean_error,
            'max_error': max_error,
            'correlation': corr,
            'p_value': p_val
        }
        
        print(f"\n📈 МЕТОД: {name}")
        if test_zeros is not None:
            print(f"{'n':<6} {'Предсказано':<15} {'Реальность':<15} {'Ошибка':<12}")
            print("-" * 50)
            for i, (pred, real, err) in enumerate(zip(predictions, test, errors)):
                print(f"{n_train + i + 1:<6} {pred:<15.6f} {real:<15.6f} {err:<12.6f}")
            print(f"\n🎯 Корреляция: r = {corr:.6f} (p = {p_val:.2e})")
            print(f"📊 Средняя ошибка: {mean_error:.6f}")
            print(f"📊 Макс. ошибка: {max_error:.6f}")
        else:
            print(f"{'n':<6} {'Предсказано':<15}")
            print("-" * 25)
            for i, pred in enumerate(predictions):
                print(f"{n_train + i + 1:<6} {pred:<15.6f}")
    
    # Сравнение методов (только если есть test)
    if test_zeros is not None:
        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ МЕТОДОВ")
        print("=" * 80)
        print(f"\n{'Метод':<35} {'Корреляция':<12} {'Ср. ошибка':<12} {'Макс. ошибка':<12}")
        print("-" * 70)
        
        for method, name in zip(methods, method_names):
            r = results[method]['correlation']
            err = results[method]['mean_error']
            max_err = results[method]['max_error']
            print(f"{name:<35} {r:<12.6f} {err:<12.6f} {max_err:<12.6f}")
        
        # Вердикт
        print("\n" + "=" * 80)
        print("ВЕРДИКТ")
        print("=" * 80)
        
        if 'spectral_extrapolation' in results:
            op_r = results['spectral_extrapolation']['correlation']
            direct_r = results['direct_fit']['correlation']
            
            if op_r > direct_r + 0.05:
                print("""
    ✅✅✅ ОПЕРАТОР РАБОТАЕТ!
    
    Спектральная экстраполяция (через оператор) 
    ЗНАЧИТЕЛЬНО точнее простой экстраполяции.
                """)
            elif op_r > direct_r:
                print("""
    👍 ОПЕРАТОР РАБОТАЕТ, НО ЭФФЕКТ НЕБОЛЬШОЙ
    
    Предсказания с оператором точнее прямой экстраполяции.
                """)
            else:
                print("""
    ⚠️ ОПЕРАТОР НЕ УЛУЧШАЕТ ПРЕДСКАЗАНИЯ
    
    Прямая экстраполяция работает так же хорошо или лучше.
                """)
    
    # Визуализация (если есть test)
    if test_zeros is not None:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # График 1: Предсказания vs реальность
        ax1 = axes[0]
        n_range = np.arange(n_train + 1, n_train + n_test + 1)
        
        ax1.plot(n_range, test, 'ko-', label='Реальные нули', markersize=8, linewidth=2)
        
        colors = {'spectral_extrapolation': 'blue', 'direct_fit': 'red', 'weyl_extrapolation': 'green'}
        labels = {'spectral_extrapolation': 'Оператор', 'direct_fit': 'Прямая экстраполяция', 'weyl_extrapolation': 'Вейль'}
        
        for method in methods:
            if method in results:
                ax1.plot(n_range, results[method]['predictions'], 
                        's-', label=labels.get(method, method), 
                        color=colors.get(method, 'gray'), 
                        markersize=6, linewidth=1.5)
        
        ax1.set_xlabel('Номер нуля n')
        ax1.set_ylabel('γ_n')
        ax1.set_title('Сравнение предсказаний')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # График 2: Ошибки
        ax2 = axes[1]
        width = 0.35
        x = np.arange(n_test)
        
        plot_methods = [m for m in methods if m in results and results[m]['errors'] is not None]
        for idx, method in enumerate(plot_methods):
            offset = width * (idx - len(plot_methods)/2 + 0.5)
            ax2.bar(x + offset, results[method]['errors'], width, 
                    label=labels.get(method, method), 
                    color=colors.get(method, 'gray'), 
                    alpha=0.7)
        
        ax2.set_xlabel('Индекс в тесте')
        ax2.set_ylabel('Абсолютная ошибка')
        ax2.set_title('Сравнение ошибок')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{n_train + i + 1}' for i in range(n_test)])
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('honest_operator_test.png', dpi=150)
        print("\n📈 График сохранён как 'honest_operator_test.png'")
        plt.show()
    
    return results, train, test


def run_cross_validation(n_splits=5, train_size=6, test_size=6):
    """
    Кросс-валидация для оценки стабильности.
    
    Разбивает первые (n_splits * (train_size + test_size)) нулей
    на n_splits блоков и проверяет предсказания.
    """
    print("\n" + "=" * 80)
    print("КРОСС-ВАЛИДАЦИЯ ОПЕРАТОРА")
    print("=" * 80)
    
    zeros = np.loadtxt('zeros_2M.txt')
    block_size = train_size + test_size
    n_blocks = min(n_splits, len(zeros) // block_size)
    
    correlations = []
    mean_errors = []
    
    hp = HonestHilbertPolyaOperator()
    
    for block in range(n_blocks):
        start = block * block_size
        train = zeros[start:start + train_size]
        test = zeros[start + train_size:start + block_size]
        
        hp.fit(train)
        predictions = hp.predict_next(test_size, method='weyl_extrapolation')
        
        corr, _ = pearsonr(predictions, test)
        mean_err = np.mean(np.abs(predictions - test))
        
        correlations.append(corr)
        mean_errors.append(mean_err)
        
        print(f"\nБлок {block + 1}: n = {start + 1}..{start + block_size}")
        print(f"  Корреляция: {corr:.6f}")
        print(f"  Средняя ошибка: {mean_err:.6f}")
    
    print("\n" + "=" * 80)
    print("ИТОГИ КРОСС-ВАЛИДАЦИИ")
    print("=" * 80)
    print(f"\nСредняя корреляция: {np.mean(correlations):.6f} ± {np.std(correlations):.6f}")
    print(f"Средняя ошибка: {np.mean(mean_errors):.6f} ± {np.std(mean_errors):.6f}")
    print(f"Доля положительных корреляций: {np.sum(np.array(correlations) > 0) / len(correlations):.1%}")
    
    return correlations, mean_errors


if __name__ == "__main__":
    # Запуск честного теста
    results, train, test = run_honest_test(n_train=8, n_test=4)
    
    # Кросс-валидация (опционально, долго)
    # run_cross_validation(n_splits=5, train_size=6, test_size=6)

    # ЗАПУСТИТЕ ЭТОТ КОД ПРЯМО СЕЙЧАС
import numpy as np
from Teor10 import run_honest_test  # импортируем из вашего файла

# Загружаем нули
zeros = np.loadtxt('zeros_2M.txt')

print("\n" + "=" * 80)
print("ТЕСТ №1: ПЕРВЫЕ 12 НУЛЕЙ")
print("=" * 80)

# Тест на первых 12 нулях (8 train, 4 test)
results1, train1, test1 = run_honest_test(
    train_zeros=zeros[:8],     # γ₁..γ₈
    test_zeros=zeros[8:12]      # γ₉..γ₁₂
)

print("\n" + "=" * 80)
print("ТЕСТ №2: ДРУГИЕ НУЛИ (γ₁₀₀..γ₁₁₁)")
print("=" * 80)

# Тест на других нулях
results2, train2, test2 = run_honest_test(
    train_zeros=zeros[99:107],   # γ₁₀₀..γ₁₀₇  
    test_zeros=zeros[107:111]     # γ₁₀₈..γ₁₁₁
)

print("\n" + "=" * 80)
print("ТЕСТ №3: СЛУЧАЙНЫЙ БЛОК")
print("=" * 80)

# Случайный блок для проверки универсальности
np.random.seed(42)
start = np.random.randint(0, len(zeros) - 20)
results3, train3, test3 = run_honest_test(
    train_zeros=zeros[start:start+8],
    test_zeros=zeros[start+8:start+12]
)