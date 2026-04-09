import numpy as np
import matplotlib.pyplot as plt
from csv_routines import read_dataset

def plot_side_by_side_hist(before_calib, after_calib, x_lim=0, y_lim=0, step_size=0):
    # 1. Настройка параметров бинов
    x_len = np.max(before_calib) * 1.05
    bin_step = 0.2
    bins = np.arange(0, x_len, bin_step)
    bin_centers = bins[:-1] + bin_step / 2 * 0.9
    
    # 2. Расчет частот (гистограммы)
    counts_before, _ = np.histogram(before_calib, bins=bins)
    counts_after, _ = np.histogram(after_calib, bins=bins)
    
    # 3. Расчет статистик для легенды
    mean_b, std_b = np.mean(before_calib), np.std(before_calib)
    mean_a, std_a = np.mean(after_calib), np.std(after_calib)

    fig, ax = plt.subplots(figsize=(24, 14))

    # Ширина одного столбца (чуть меньше половины шага бина, чтобы был зазор)
    width = bin_step/2 

    # 4. Отрисовка столбцов рядом
    # Синие (After) — смещаем влево
    rects1 = ax.bar(bin_centers - width/2, counts_after, width, 
                    color='#9999ff', edgecolor='blue', alpha=0.8,
                    label=f'После калибровки: среднее = {mean_a:.3f} mm, стандартное отклонение = {std_a:.3f} mm')
    
    # Красные (Before) — смещаем вправо
    rects2 = ax.bar(bin_centers + width/2, counts_before, width, 
                    color='#ff9999', edgecolor='red', alpha=0.8,
                    label=f'До калибровки: среднее = {mean_b:.3f} mm, стандартное отклонение = {std_b:.3f} mm')

    # 5. Оформление осей и сетки
    ax.set_xlabel('Ошибка позиционирования (мм)', fontsize=16, labelpad=10)
    ax.set_ylabel('Число измерений', fontsize=16)
    
    # Настройка делений оси X (каждые 0.2 как на картинке)
    ax.set_xticks(np.arange(0, x_len, 0.4))
    ax.set_xlim(-0.1, x_len)
    
    # Настройка сетки (основная и вспомогательная)
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True) # Сетка под графиком
    
    # Добавляем мелкую сетку для эффекта "клетки"
    ax.set_xticks(np.arange(0, 3.5, 0.1), minor=True)
    ax.set_yticks(np.arange(0, 7, 1), minor=True)
    ax.grid(True, which='minor', color='lightgray', linestyle='-', linewidth=0.3)

    # 6. Легенда (в рамке, вверху справа)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', 
              facecolor='white', fontsize=15, borderpad=0.8)

    plt.tight_layout()
    plt.show()


def plot_dh_comparison(before, after):
    # Создаем индексы для 24 параметров (например, Link 1, Link 2...)
    n_params = len(before)
    indices = np.arange(0, n_params)
    
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

# --- Тестовые данные ---
# Создаем выборки, близкие к вашим значениям
def main():
    data = read_dataset("results/ARM95/test_result.csv", ['q1',	'q2',	'q3',	'q4',	'q5',	'q6',	
                                                        'd_nom',	'd_est',	'delta'])
    data_before = data[:, 6]
    data_after = data[:, 7]


    # Запускаем функцию
    plot_side_by_side_hist(data_before, data_after)

    # --- Пример данных ---
    np.random.seed(10)
    # Допустим, начальные ошибки параметров звеньев около 1-3 мм
    errors_before = np.random.normal(2.0, 0.5, 24) 
    # После калибровки ошибки упали до 0.1-0.5 мм
    errors_after = np.random.normal(0.4, 0.1, 24)

    plot_parameter_comparison(errors_before, errors_after)

if __name__ == "__main__":
    main()