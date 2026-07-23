import numpy as np
import matplotlib.pyplot as plt

def generate_sensor_test_trajectory(sensor_pos: list, radius: float = 400.0, plot: bool = False) -> np.ndarray:
    """
    Генерирует 162 точки TCP для оценки критического угла датчика.
    sensor_pos: [X, Y, Z] координаты датчика.
    radius: базовая дистанция вытягивания (рекомендовано 400 мм).
    plot: флаг отрисовки 3D-графика.
    """
    sensor_pos = np.array(sensor_pos, dtype=float)
    
    # Диапазон: от 20 до 162 градусов с шагом 2 (одна большая дуга = 142 градуса)
    angles_deg = np.arange(20, 162, 2)
    angles_rad = np.radians(angles_deg)
    
    arc_xz = []
    arc_yz = []

    for a in angles_rad:
        # Дуга в плоскости XZ (Базовый вектор смотрит по оси X)
        # Угол 'a' отклоняет вектор вверх и вниз по оси Z
        x1 = radius * np.cos(a)
        y1 = 0.0
        z1 = radius * np.sin(a)
        arc_xz.append(sensor_pos + np.array([x1, y1, z1]))
        
    for a in angles_rad:
        # Дуга в плоскости YZ (Базовый вектор смотрит по оси Y)
        # Угол 'a' отклоняет вектор вверх и вниз по оси Z
        x2 = 0.0
        y2 = radius * np.cos(a)
        z2 = radius * np.sin(a)
        arc_yz.append(sensor_pos + np.array([x2, y2, z2]))

    arc_xz_np = np.array(arc_xz)
    arc_yz_np = np.array(arc_yz)

    # Объединение в единый датасет (162 точки: 81 + 81)
    trajectory = np.vstack((arc_xz_np, arc_yz_np))

    if plot:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Отрисовка дуг
        ax.plot(arc_xz_np[:, 0], arc_xz_np[:, 1], arc_xz_np[:, 2], 'b.-', label='Дуга XZ (База = X)')
        ax.plot(arc_yz_np[:, 0], arc_yz_np[:, 1], arc_yz_np[:, 2], 'g.-', label='Дуга YZ (База = Y)')

        # Отрисовка датчика и нулевых точек (0 градусов)
        ax.scatter(*sensor_pos, color='black', s=100, marker='s', label='Датчик')
        ax.scatter(*(sensor_pos + [radius, 0, 0]), color='blue', s=150, marker='*', label='0° для XZ', zorder=5)
        ax.scatter(*(sensor_pos + [0, radius, 0]), color='green', s=150, marker='*', label='0° для YZ', zorder=5)

        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.set_zlabel('Z, мм')
        ax.set_title(f'Анализ перегиба троса (Смещение по Z). L_zero = {radius} мм')
        ax.legend()

        # Выравнивание масштабов осей для сохранения пропорций полусферы
        max_range = radius * 1.1
        ax.set_xlim(sensor_pos[0] - max_range, sensor_pos[0] + max_range)
        ax.set_ylim(sensor_pos[1] - max_range, sensor_pos[1] + max_range)
        ax.set_zlim(sensor_pos[2] - max_range, sensor_pos[2] + max_range)
        ax.set_box_aspect((1, 1, 1))

        plt.show()

    return trajectory

if __name__ == "__main__":
    # Пример вызова с отрисовкой
    points = generate_sensor_test_trajectory(sensor_pos=[0.0, 0.0, 0.0], radius=400.0, plot=True)
    print(f"Сгенерировано точек: {len(points)}")