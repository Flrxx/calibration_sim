import numpy as np
import matplotlib.pyplot as plt

def plot_parameter_comparison(before, after):
    # Создаем индексы для 24 параметров (например, Link 1, Link 2...)
    n_params = len(before)
    indices = np.arange(1, n_params + 1)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    width = 0.35  # Ширина столбца

    # Строим столбцы для каждого параметра
    rects1 = ax.bar(indices - width/2, before, width, 
                    label='Before Calibration', color='#ff9999', edgecolor='red', alpha=0.8)
    rects2 = ax.bar(indices + width/2, after, width, 
                    label='After Calibration', color='#9999ff', edgecolor='blue', alpha=0.8)

    # Оформление осей
    ax.set_xlabel('Parameter Index (Link / Joint #)', fontsize=12)
    ax.set_ylabel('Error Value (mm / rad)', fontsize=12)
    ax.set_title('Robot Parameters Calibration Results (24 Parameters)', fontsize=14)
    
    # Настройка делений (чтобы каждое число от 1 до 24 было подписано)
    ax.set_xticks(indices)
    ax.set_xticklabels([f'{i}' for i in indices])

    # Сетка как в оригинале (клеточками)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Добавляем легенду
    ax.legend(loc='upper right', frameon=True, edgecolor='black')

    # Дополнительно: выведем общую точность (RMS) в углу
    rms_before = np.sqrt(np.mean(before**2))
    rms_after = np.sqrt(np.mean(after**2))
    
    stats_text = (f'RMS Before: {rms_before:.3f}\n'
                  f'RMS After: {rms_after:.3f}')
    
    # Размещаем текст со статистикой в рамке
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# --- Пример данных ---
np.random.seed(10)
# Допустим, начальные ошибки параметров звеньев около 1-3 мм
errors_before = np.random.normal(2.0, 0.5, 24) 
# После калибровки ошибки упали до 0.1-0.5 мм
errors_after = np.random.normal(0.4, 0.1, 24)

plot_parameter_comparison(errors_before, errors_after)