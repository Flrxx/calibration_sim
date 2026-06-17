import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from csv_routines import read_dataset # Раскомментируйте у себя

def plot_side_by_side_hist(before_calib, after_calib, x_lim=0, y_lim=0, step_size=0):
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

    # 4. Отрисовка столбцов (ЧБ контраст и штриховка)
    ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#444444', edgecolor='black', alpha=1.0,
                    label=f'После калибровки: среднее = {mean_a:.3f} мм, ст. откл. = {std_a:.3f} мм')
    
    ax.bar(bin_centers + width/2, counts_before, width, 
                    color='white', edgecolor='black', hatch='///', alpha=1.0,
                    label=f'До калибровки: среднее = {mean_b:.3f} мм, ст. откл. = {std_b:.3f} мм')

    # 5. Отрисовка нормального распределения (черные сплошная и пунктирная линии)
    x_axis = np.linspace(0, x_max, 500)
    line_before = norm.pdf(x_axis, mean_b, std_b) * len(before_calib) * bin_step
    line_after = norm.pdf(x_axis, mean_a, std_a) * len(after_calib) * bin_step
    
    ax.plot(x_axis, line_before, color='black', linestyle='--', linewidth=3)
    ax.plot(x_axis, line_after, color='black', linestyle='-', linewidth=3)

    # 6. Оформление осей и сетки
    ax.set_xlabel('Ошибка позиционирования (мм)', fontsize=12, labelpad=15)
    ax.set_ylabel('Число измерений', fontsize=12)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    actual_x_max = x_lim if x_lim > 0 else x_max
    ax.set_xticks(np.arange(0, actual_x_max, 0.4))
    ax.set_xlim(-0.1, actual_x_max)
    if y_lim > 0: ax.set_ylim(0, y_lim)
    
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3)

    # 7. Легенда
    ax.legend(loc='upper right', frameon=True, edgecolor='black', 
              facecolor='white', fontsize=12, borderpad=0.8)
    
    plt.tight_layout()
    plt.show()

def plot_dh_comparison(before, after):
    param_names = ["alpha_6", "a_6", "d_6", "a_5", "theta_6", "alpha_5",   
                 "theta_5", "a_4", "d_5", "alpha_4", "beta_3", "theta_4", "alpha_3", "a_3", "theta_3", 
                 "beta_2", "alpha_2", "a_2", "d_4", "theta_2", "alpha_1", "a_1",      
                 "d_1", "theta_1"]
    n_params = len(before)
    indices = np.arange(n_params)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.35 

    # Столбцы для ЧБ печати: текстурированный белый ("До") и залитый темно-серый ("После")
    ax.bar(indices - width/2, before, width, 
                    label='До калибровки', color='white', edgecolor='black', hatch='\\\\\\')
    ax.bar(indices + width/2, after, width, 
                    label='После калибровки', color='#444444', edgecolor='black')

    ax.set_xlabel('Параметры (Звено / Сочленение)', fontsize=14)
    ax.set_ylabel('Значение ошибки (мм / градус)', fontsize=14)
    ax.set_title('Результаты калибровки параметров робота (24 параметра)', fontsize=16)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    rms_before = np.sqrt(np.mean(before**2))
    rms_after = np.sqrt(np.mean(after**2))
    
    stats_text = (f'СКО (RMS) До: {rms_before:.3f}\n'
                  f'СКО (RMS) После: {rms_after:.3f}')
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=12)

    plt.tight_layout()
    plt.show()

def filter_outliers_and_report(col_before, col_after, axis_name):
    """
    Фильтрует выбросы независимо для 'before' и 'after' для конкретной оси.
    Выводит в терминал количество удаленных строк.
    """
    def get_clean_mask(data):
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        threshold = median + 10 * mad if mad > 0 else median * 10
        return data <= threshold

    mask_b = get_clean_mask(col_before)
    mask_a = get_clean_mask(col_after)
    
    deleted_b = len(col_before) - np.sum(mask_b)
    deleted_a = len(col_after) - np.sum(mask_a)
    
    print(f"Ось [{axis_name}]:")
    print(f"  -> Удалено выбросов 'До калибровки': {deleted_b} строк(и)")
    print(f"  -> Удалено выбросов 'После калибровки': {deleted_a} строк(и)")
    print("-" * 50)
    
    return col_before[mask_b], col_after[mask_a]

def plot_single_hist_window(before_calib, after_calib, title, x_lim=0, y_lim=0):
    # 1. Настройка параметров бинов
    x_max = np.max(before_calib) * 1.05 if np.max(before_calib) > 0 else 1.0
    bin_step = 0.2
    bins = np.arange(0, x_max + bin_step, bin_step)
    bin_centers = bins[:-1] + bin_step / 2
    
    # 2. Расчет частот (гистограммы)
    counts_before, _ = np.histogram(before_calib, bins=bins)
    counts_after, _ = np.histogram(after_calib, bins=bins)
    
    # 3. Расчет статистик для легенды
    mean_b, std_b = np.mean(before_calib), np.std(before_calib)
    mean_a, std_a = np.mean(after_calib), np.std(after_calib)

    fig, ax = plt.subplots(figsize=(16, 8), num=title)
    ax.set_title(f"Распределение ошибок {title}", fontsize=14, pad=15)

    # Ширина одного столбца
    width = bin_step / 2.5 

    # 4. Отрисовка столбцов (ЧБ контраст и штриховка)
    ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#444444', edgecolor='black', alpha=1.0,
                    label=f'После калибровки: среднее = {mean_a:.3f}, ст. откл. = {std_a:.3f}')
    
    ax.bar(bin_centers + width/2, counts_before, width, 
                    color='white', edgecolor='black', hatch='///', alpha=1.0,
                    label=f'До калибровки: среднее = {mean_b:.3f}, ст. откл. = {std_b:.3f}')

    # 5. Отрисовка нормального распределения (черные сплошная и пунктирная линии)
    x_axis = np.linspace(0, x_max, 500)
    std_b_safe = std_b if std_b > 0 else 1e-6
    std_a_safe = std_a if std_a > 0 else 1e-6
    
    line_before = norm.pdf(x_axis, mean_b, std_b_safe) * len(before_calib) * bin_step
    line_after = norm.pdf(x_axis, mean_a, std_a_safe) * len(after_calib) * bin_step
    
    ax.plot(x_axis, line_before, color='black', linestyle='--', linewidth=3)
    ax.plot(x_axis, line_after, color='black', linestyle='-', linewidth=3)

    # 6. Оформление осей и сетки
    ax.set_xlabel('Ошибка позиционирования', fontsize=12, labelpad=15)
    ax.set_ylabel('Число измерений', fontsize=12)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    actual_x_max = x_lim if x_lim > 0 else x_max
    ax.set_xticks(np.arange(0, actual_x_max, 0.4))
    ax.set_xlim(-0.1, actual_x_max)
    if y_lim > 0: ax.set_ylim(0, y_lim)
    
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3)

    ax.legend(loc='upper right', frameon=True, edgecolor='black', 
              facecolor='white', fontsize=12, borderpad=0.8)
    
    plt.tight_layout()

def plot_all_6_axes(matrix_before, matrix_after):
    """
    matrix_before: np.array формы (N, 6)
    matrix_after: np.array формы (N, 6)
    """
    window_titles = ['x', 'y', 'z', 'alpha', 'beta', 'gamma']
    
    print("--- ОТЧЕТ ОБ ОЧИСТКЕ ДАННЫХ ---")
    for i in range(6):
        col_before = matrix_before[:, i]
        col_after = matrix_after[:, i]
        title = window_titles[i]
        
        # Фильтруем данные перед отправкой на график
        clean_before, clean_after = filter_outliers_and_report(col_before, col_after, title)
        
        # Строим график по очищенным данным
        plot_single_hist_window(clean_before, clean_after, title=title)
    
    plt.show()
    
# Для реального эксперимента
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

    # 4. Отрисовка столбцов (ЧБ контраст и текстурная штриховка)
    ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#444444', edgecolor='black', alpha=1.0,
                    label=f'После калибровки: среднее = {mean_a:.3f} мм, ст. откл. = {std_a:.3f} мм')
    
    ax.bar(bin_centers + width/2, counts_before, width, 
                    color='white', edgecolor='black', hatch='///', alpha=1.0,
                    label=f'До калибровки: среднее = {mean_b:.3f} мм, ст. откл. = {std_b:.3f} мм')

    # 5. Отрисовка нормального распределения (черные сплошная и пунктирная линии поверх столбцов)
    x_axis = np.linspace(0, x_max, 500)
    line_before = norm.pdf(x_axis, mean_b, std_b) * len(before_calib) * bin_step
    line_after = norm.pdf(x_axis, mean_a, std_a) * len(after_calib) * bin_step
    
    ax.plot(x_axis, line_before, color='black', linestyle='--', linewidth=3)
    ax.plot(x_axis, line_after, color='black', linestyle='-', linewidth=3)

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
    

def main():
    # Тестовые данные для демонстрации одной гистограммы и 24 параметров
    np.random.seed(42)
    data_before = np.random.normal(2.0, 0.6, 100)
    data_after = np.random.normal(0.4, 0.15, 100)

    # 1. График гистограмм
    plot_side_by_side_hist(data_before, data_after)

    # 2. График сравнения 24 параметров
    param_names = ["a_6", "d_6", "alpha_6", "a_5", "theta_6",
                   "theta_5", "a_4", "alpha_5", "d_5", "alpha_4", "beta_3", "theta_4", "alpha_3", "a_3", "theta_3", 
                   "beta_2", "alpha_2", "a_2", "d_4", "theta_2", "alpha_1", "a_1",      
                   "d_1", "theta_1"]
    errors_before = np.random.normal(1.5, 0.4, 24) 
    errors_after = np.random.normal(0.2, 0.05, 24)

    plot_dh_comparison(param_names, errors_before, errors_after)

    # 3. Демонстрация 6 окон с фильтрацией выбросов
    mock_before = np.abs(np.random.normal(loc=1.0, scale=0.2, size=(200, 6)))
    mock_after = np.abs(np.random.normal(loc=0.2, scale=0.05, size=(200, 6)))
    mock_before[10, 0] = 50.0  # Аномалия для теста фильтра
    mock_after[85, 2] = 35.0   # Аномалия для теста фильтра

    plot_all_6_axes(mock_before, mock_after)

if __name__ == "__main__":
    main()