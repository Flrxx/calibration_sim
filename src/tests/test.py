import numpy as np
from scipy.optimize import minimize

# 1. Ваш "Черный ящик"
def user_function(x):
    # Пример: выход из 5 чисел
    # Допустим, мы хотим максимизировать 3-й элемент (индекс 2)
    return np.array([
        x[0] + x[1],       # Должен быть 0
        x[0] - x[2],       # Должен быть 0
        x[0] * x[3],       # ХОТИМ МАКСИМИЗИРОВАТЬ ЭТО
        x[4]**2 - 1,       # Должен быть 0
        x[5] + x[0]        # Должен быть 0
    ])

# 2. Настройка задачи
target_index = 2   # Индекс элемента, который нужно максимизировать
penalty_weight = 1000.0 # Насколько важно, чтобы остальные были нулями

def objective(x):
    y = user_function(x)
    
    # Значение, которое максимизируем (берем с минусом для minimize)
    target_val = y[target_index]
    
    # Остальные значения (выкидываем target_index)
    rest_vals = np.delete(y, target_index)
    
    # Считаем ошибку: 
    # (минус цель) + (штраф * сумма квадратов остальных)
    loss = -target_val + penalty_weight * np.sum(rest_vals**2)
    return loss

# 3. Запуск
x0 = np.random.rand(6) # Случайный старт
# method='BFGS' или 'Nelder-Mead' хорошо подходят для таких задач
res = minimize(objective, x0, method='BFGS')

print("Вход x:", res.x)
print("Выход y:", user_function(res.x))