from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
california_housing = fetch_california_housing()
df = pd.DataFrame(data=california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target

# Шаг 2: Предварительная обработка данных
X = df[['MedInc']].values  # Используем средний доход как признак
y = df['MedHouseVal'].values

# Нормализация признаков
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Добавление bias
X_bias = np.hstack([np.ones((X_normalized.shape[0], 1)), X_normalized])

# Шаг 3: Разделение данных
train_size = int(0.8 * X_bias.shape[0])
X_train = X_bias[:train_size]
y_train = y[:train_size]
X_test = X_bias[train_size:]
y_test = y[train_size:]

# Шаг 4: Реализация модели линейной регрессии

# Инициализация весов
np.random.seed(42)
weights = np.random.randn(X_train.shape[1])

# Гипотеза
def predict(X, weights):
    return np.dot(X, weights)

# Функция потерь (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Градиент
def compute_gradient(X, y_true, y_pred):
    m = X.shape[0]
    gradient = (-2/m) * np.dot(X.T, (y_true - y_pred))
    return gradient

# Обновление весов
def update_weights(weights, gradient, learning_rate):
    return weights - learning_rate * gradient

# Параметры градиентного спуска
learning_rate = 0.01
epochs = 400
tolerance = 1e-6

# Хранение истории потерь
loss_history = []

for epoch in range(epochs):
    # Шаг 1: Предсказания
    y_pred = predict(X_train, weights)
    
    # Шаг 2: Вычисление потерь
    loss = mean_squared_error(y_train, y_pred)
    loss_history.append(loss)
    
    # Шаг 3: Вычисление градиента
    gradient = compute_gradient(X_train, y_train, y_pred)
    
    # Шаг 4: Обновление весов
    weights = update_weights(weights, gradient, learning_rate)
    
    # Шаг 5: Проверка условия остановки
    if epoch > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
        print(f"Остановка на эпохе {epoch}, изменение потерь менее {tolerance}")
        break

# Шаг 5: Оценка модели
def predict_test(X, weights):
    return np.dot(X, weights)

y_test_pred = predict_test(X_test, weights)

# Вычисление метрик
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = np.mean(np.abs(y_test - y_test_pred))

print(f"RMSE на тестовом наборе: {rmse:.2f}")
print(f"MAE на тестовом наборе: {mae:.2f}")
print(f"Количество эпох - {epoch}")

# Шаг 6: Визуализация потерь
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Потери (MSE)')
plt.xlabel('Эпохи')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('График потерь во время обучения')
plt.legend()
plt.grid(True)
plt.show()

# Шаг 7: Визуализация предсказаний
plt.figure(figsize=(8, 8))
plt.scatter(X_test[:, 1], y_test_pred, color='blue', edgecolor='k', alpha=0.7, label='Предсказанные значения')
plt.scatter(X_test[:, 1], y_test, color='green', edgecolor='k', alpha=0.3, label='Истинные значения')
plt.xlabel("MedInc (Средний доход) [нормализовано]")
plt.ylabel("MedHouseVal (Средняя стоимость дома в $100k)")
plt.title("Линейная регрессия: Истинные vs Предсказанные значения")
plt.legend()
plt.grid(True)
plt.show()
