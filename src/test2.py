import matplotlib.pyplot as plt
import numpy as np
from csv_routines import read_dataset
import hayati_model
import json
import argparse
from math_routines import DEG

def plot_3d_points_by_sign(
    points, vector, title="3D визуализация точек", labels=("X", "Y", "Z")
):
    """Визуализирует 3D точки разным цветом в зависимости от знака вектора.

    Parameters:
    ----------
    points : np.ndarray
        Массив точек формы (N, 3).
    vector : np.ndarray
        Вектор значений формы (N,).
    title : str
        Заголовок графика.
    labels : tuple
        Названия осей (X, Y, Z).
    """
    # 1. Создаем булевые маски
    pos_mask = vector >= 0
    neg_mask = vector < 0

    # 2. Инициализируем 3D график
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    # 3. Отрисовка положительных точек (зеленые)
    if np.any(pos_mask):
        ax.scatter(
            points[pos_mask, 0],
            points[pos_mask, 1],
            points[pos_mask, 2],
            c="green",
            label=f"Положительные (>= 0): {np.sum(pos_mask)} шт.",
            alpha=0.8,
            edgecolors="w",
            s=40,
        )

    # 4. Отрисовка отрицательных точек (красные)
    if np.any(neg_mask):
        ax.scatter(
            points[neg_mask, 0],
            points[neg_mask, 1],
            points[neg_mask, 2],
            c="red",
            label=f"Отрицательные (< 0): {np.sum(neg_mask)} шт.",
            alpha=0.8,
            edgecolors="w",
            s=40,
        )

    # 5. Настройка окружения графика
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    ax.grid(True)
    ax.legend(loc="upper left")

    plt.show()


# === Пример использования ===
if __name__ == "__main__":
    # # Генерируем тестовые данные
    # n_samples = 150
    # test_points = np.random.uniform(-5, 5, size=(n_samples, 3))
    # test_vector = np.random.uniform(-2, 2, size=n_samples)
    data = read_dataset('results/ARM95/validation_results_real.csv', ["q1", "q2", "q3", "q4", "q5", "q6", "d_nom", "d_est", "d_encoder", "diff"])
    data = data[:-1]
    points = np.zeros(shape=(data.shape[0], 3))
    
    with open("src/config/ARM95_calibration.json", 'r') as config_file:
        config = json.load(config_file)
    model = hayati_model.HayatiModel(config)
    for i in range(data.shape[0]):
        q = data[i, :6] * DEG
        T = model.get_transition_matrix(q, "nominal")
        points[i] = T[:3, 3]
        print(points[i])

    # Вызываем функцию
    plot_3d_points_by_sign(
        points=points,
        vector=data[:, 9],
        title="Тест разделения точек по знаку",
        labels=("Ось X", "Ось Y", "Ось Z"),
    )