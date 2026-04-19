import numpy as np
from scipy.stats import gamma
import mpmath as mp

mp.mp.dps = 50

def siegel_theta(t):
    return float(mp.siegeltheta(t))

def get_gram_class(t):
    gram_index = int(round(siegel_theta(t) / mp.pi))
    return gram_index % 12

# --- ЗАГРУЗКА ДАННЫХ (как раньше) ---
def load_odlyzko(filename):
    base = 267653395647.0
    zeros = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Values'): continue
            try:
                zeros.append(base + float(line))
            except: pass
    return np.array(zeros)

zeros = load_odlyzko('zero_10k_10^12.txt')
gram_classes = np.array([get_gram_class(t) for t in zeros])

# --- ЧЕСТНЫЙ АНАЛИЗ ПОТОКОВ ---
stream_k = []

print("Поток | N    | mean_Δ      | k (MLE)")
print("-" * 40)

for c in range(12):
    # Выбираем нули ОДНОГО класса
    stream_zeros = zeros[gram_classes == c]
    if len(stream_zeros) < 10:
        continue
        
    # Интервалы МЕЖДУ НУЛЯМИ ЭТОГО КЛАССА
    intervals = np.diff(stream_zeros)
    
    # Нормируем на СРЕДНЕЕ ЭТОГО ЖЕ КЛАССА
    mean_int = np.mean(intervals)
    norm_intervals = intervals / mean_int
    
    # Оцениваем k гамма-распределения (MLE)
    # mean = 1.0, scale = 1/k
    k_hat, _, _ = gamma.fit(norm_intervals, floc=0)
    stream_k.append(k_hat)
    
    print(f"{c:4d} | {len(stream_zeros):4d} | {mean_int:10.6f} | {k_hat:.4f}")

# --- РЕЗУЛЬТАТ ---
k_mean = np.mean(stream_k)
print("\n" + "=" * 40)
print(f"НАБЛЮДАЕМОЕ СРЕДНЕЕ k = {k_mean:.4f}")
print(f"ТЕОРЕТИЧЕСКИЙ ПРЕДЕЛ 12/10 = 1.2000")
print(f"ОШИБКА = {abs(k_mean - 1.200) / 1.200 * 100:.2f}%")