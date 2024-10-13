import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# URL de la API
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'

params = {
    'vs_currency': 'usd',
    'days': '30', 
    'interval': 'daily' 
}

response = requests.get(url, params=params)
data = response.json()

if response.status_code == 200:
    print("Datos obtenidos correctamente.")
else:
    print("Error en la solicitud:", response.status_code)


prices = data['prices']

df = pd.DataFrame(prices, columns=['timestamp', 'price'])

df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('date', inplace=True)

df.drop('timestamp', axis=1, inplace=True)

print(df.head())

# Visualizar los datos
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='price')
plt.title('Precio de Bitcoin en los últimos 30 días')
plt.xlabel('Fecha')
plt.ylabel('Precio (USD)')
plt.grid()
plt.show()

# Preprocesar los datos
# Normalizar los precios
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['price']].values)

#conjuntos de entrenamiento y prueba
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 5 
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

# conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Red Neuronal
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Capa de salida

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Realizar predicciones
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertir la normalización para obtener los precios originales
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calcular el RMSE
import math
from sklearn.metrics import mean_squared_error
train_score = math.sqrt(mean_squared_error(y_train, train_predict[:,0]))
test_score = math.sqrt(mean_squared_error(y_test, test_predict[:,0]))
print(f'RMSE en datos de entrenamiento: {train_score}')
print(f'RMSE en datos de prueba: {test_score}')

# Visualizar los resultados
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

# Hacer las predicciones en el conjunto de prueba
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

# Visualizar los resultados
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='Precio Real')
plt.plot(train_predict_plot, label='Predicción Entrenamiento')
plt.plot(test_predict_plot, label='Predicción Prueba')
plt.title('Predicción del Precio de Bitcoin')
plt.xlabel('Días')
plt.ylabel('Precio (USD)')
plt.legend()
plt.show()
