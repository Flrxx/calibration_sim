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

    # 4. Отрисовка столбцов (прозрачность alpha уменьшена до 0.5 для видимости линий)
    ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#9999ff', edgecolor='blue', alpha=0.5,
                    label=f'После калибровки: среднее = {mean_a:.3f} мм, ст. откл. = {std_a:.3f} мм')
    
    ax.bar(bin_centers + width/2, counts_before, width, 
                    color='#ff9999', edgecolor='red', alpha=0.5,
                    label=f'До калибровки: среднее = {mean_b:.3f} мм, ст. откл. = {std_b:.3f} мм')

    # 5. Отрисовка нормального распределения
    x_axis = np.linspace(0, x_max, 500)
    # Масштабируем PDF под высоту гистограммы: PDF * кол-во_точек * ширина_бина
    line_before = norm.pdf(x_axis, mean_b, std_b) * len(before_calib) * bin_step
    line_after = norm.pdf(x_axis, mean_a, std_a) * len(after_calib) * bin_step
    
    ax.plot(x_axis, line_before, color='red', linestyle='-', linewidth=3, alpha=0.9)
    ax.plot(x_axis, line_after, color='blue', linestyle='-', linewidth=3, alpha=0.9)

    # 6. Оформление осей и сетки
    ax.set_xlabel('Ошибка позиционирования (мм)', fontsize=12, labelpad=15)
    ax.set_ylabel('Число измерений', fontsize=12)
    
    # Увеличение шрифта самих чисел на осях
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    actual_x_max = x_lim if x_lim > 0 else x_max
    ax.set_xticks(np.arange(0, actual_x_max, 0.4))
    ax.set_xlim(-0.1, actual_x_max)
    if y_lim > 0: ax.set_ylim(0, y_lim)
    
    # Сетка
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Мелкая сетка (миллиметровка)
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3)

    # 7. Легенда
    ax.legend(loc='upper right', frameon=True, edgecolor='black', 
              facecolor='white', fontsize=12, borderpad=0.8)
    

    plt.tight_layout()
    plt.show()

def plot_dh_comparison(before, after):
    param_names = ["a_6", "d_6", "alpha_6", "a_5", "theta_6",
                 "theta_5", "a_4", "alpha_5", "d_5", "alpha_4", "beta_3", "theta_4", "alpha_3", "a_3", "theta_3", 
                 "beta_2", "alpha_2", "a_2", "d_4", "theta_2", "alpha_1", "a_1",      
                  "d_1", "theta_1"]
    n_params = len(before)
    indices = np.arange(n_params)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.35 

    ax.bar(indices - width/2, before, width, 
                    label='До калибровки', color='#ff9999', edgecolor='red', alpha=0.8)
    ax.bar(indices + width/2, after, width, 
                    label='После калибровки', color='#9999ff', edgecolor='blue', alpha=0.8)

    ax.set_xlabel('Параметры (Звено / Сочленение)', fontsize=14)
    ax.set_ylabel('Значение ошибки (мм / рад)', fontsize=14)
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

def main():
    # Загрузка данных (замените на свой путь)
    # data = read_dataset("results/ARM95/test_result.csv", ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'd_nom', 'd_est', 'delta'])
    # data_before = data[:, 6]
    # data_after = data[:, 7]

    # Тестовые данные для демонстрации
    np.random.seed(42)
    data_before = np.random.normal(2.0, 0.6, 100)
    data_after = np.random.normal(0.4, 0.15, 100)

    # 1. График гистограмм
    plot_side_by_side_hist(data_before, data_after)

    # 2. График сравнения 24 параметров
    param_names = [f"Парам. {i+1}" for i in range(24)]
    errors_before = np.random.normal(1.5, 0.4, 24) 
    errors_after = np.random.normal(0.2, 0.05, 24)

    plot_dh_comparison(param_names, errors_before, errors_after)

if __name__ == "__main__":
    main()