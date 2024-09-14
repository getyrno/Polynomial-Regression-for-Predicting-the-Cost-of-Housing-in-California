from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Загрузка датасета
california_housing = fetch_california_housing()

# Преобразование в DataFrame для удобства
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target

print(df.head())  # Просмотр первых нескольких строк данных

# Определение признаков (X) и целевой переменной (y)
X = df[['MedInc']]  # Используем средний доход как признак
y = df['MedHouseVal']

# Разделение данных: 80% для обучения и 20% для тестирования
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Вычисление метрик
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print(f"RMSE на тестовом наборе: {rmse:.2f}")
print(f"MAE на тестовом наборе: {mae:.2f}")

# Визуализация результатов
plt.scatter(X_test, y_pred, color='blue', edgecolor='k', alpha=0.7, label='Предсказанные значения')
plt.scatter(X_test, y_test, color='green', edgecolor='k', alpha=0.3, label='Истинные значения')
plt.xlabel("MedInc (Средний доход)")
plt.ylabel("MedHouseVal (Средняя стоимость дома в $100k)")
plt.title("Линейная регрессия: Истинные vs Предсказанные значения")
plt.legend()
plt.show()
