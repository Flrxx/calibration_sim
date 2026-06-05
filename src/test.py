import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from csv_routines import read_dataset

def plot_calibration_errors_from_data(data, x_lim=0, y_lim=0):
    """
    data: исходный двумерный np.array из CSV.
    Использует:
    - Модуль разности 7 и 9 столбцов (индексы 6 и 8) как "До калибровки"
    - Модуль разности 8 и 9 столбцов (индексы 7 и 8) как "После калибровки"
    """
    # Расчет модулей разностей (индексы смещены на -1, так как отсчет с 0)
    before_calib = np.abs(data[:, 6] - data[:, 8])
    after_calib = np.abs(data[:, 7] - data[:, 8])
    
    # 1. Настройка параметров бинов
    x_max = np.max(before_calib) * 1.05
    bin_step = 0.2
    bins = np.arange(0, x_max, bin_step)
    bin_centers = bins[:-1] + bin_step / 2
    
    # 2. Расчет частот (гистограммы)
    counts_before, _ = np.histogram(before_calib, bins=bins)
    counts_after, _ = np.histogram(after_calib, bins=bins)
    
    # 3. Расчет статистик для легенды
    mean_b, std_b = np.mean(before_calib), np.std(before_calib)
    mean_a, std_a = np.mean(after_calib), np.std(after_calib)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Ширина одного столбца
    width = bin_step / 2.5 

    # 4. Отрисовка столбцов (возвращаем синий и красный цвета)
    ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#9999ff', edgecolor='blue', alpha=0.5,
                    label=f'После калибровки: среднее = {mean_a:.3f} мм, ст. откл. = {std_a:.3f} мм')
    
    ax.bar(bin_centers + width/2, counts_before, width, 
                    color='#ff9999', edgecolor='red', alpha=0.5,
                    label=f'До калибровки: среднее = {mean_b:.3f} мм, ст. откл. = {std_b:.3f} мм')

    # 5. Отрисовка нормального распределения (синяя и красная линии поверх столбцов)
    x_axis = np.linspace(0, x_max, 500)
    line_before = norm.pdf(x_axis, mean_b, std_b) * len(before_calib) * bin_step
    line_after = norm.pdf(x_axis, mean_a, std_a) * len(after_calib) * bin_step
    
    ax.plot(x_axis, line_before, color='red', linestyle='-', linewidth=3, alpha=0.9)
    ax.plot(x_axis, line_after, color='blue', linestyle='-', linewidth=3, alpha=0.9)

    # 6. Оформление осей и сетки
    ax.set_xlabel('Ошибка позиционирования (мм)', fontsize=16, labelpad=15)
    ax.set_ylabel('Число точек', fontsize=16)
    
    # Крупный шрифт для значений на осях
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    actual_x_max = x_lim if x_lim > 0 else x_max
    ax.set_xticks(np.arange(0, actual_x_max, 0.4))
    ax.set_xlim(-0.1, actual_x_max)
    if y_lim > 0: ax.set_ylim(0, y_lim)
    
    # Сетка
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True) # Сетка под графиком
    
    # Мелкая сетка (миллиметровка)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3)

    # 7. Легенда
    ax.legend(loc='upper right', frameon=True, edgecolor='black', 
              facecolor='white', fontsize=12, borderpad=0.8)

    plt.tight_layout()
    plt.show()
    
    
data = read_dataset('results/ARM95/validation_results_real.csv', ["q1", "q2", "q3", "q4", "q5", "q6", "d_nom", "d_est", "d_encoder", "diff"])
data = data[:-1]
plot_calibration_errors_from_data(data)