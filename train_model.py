import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle

df = pd.read_csv('dse_data.csv', thousands=',')
df = df[['DATE', 'TRADING CODE', 'CLOSEP*']]
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.dropna(subset=['DATE'])
df = df.sort_values('DATE')

window_size = 90
X_list = []
y_list = []
company_indices = []  
scalers = {}  

for code, group in df.groupby('TRADING CODE'):
    group = group.sort_values('DATE')
    prices = group['CLOSEP*'].values.astype('float32')
    if len(prices) < window_size + 1:
        continue  
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    scalers[code] = scaler
    for i in range(window_size, len(prices_scaled)):
        X_list.append(prices_scaled[i-window_size:i])
        y_list.append(prices_scaled[i])
        company_indices.append(code)

X = np.array(X_list)
y = np.array(y_list)
X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(GRU(25, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
early_stop = EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_val, y_val), callbacks=[early_stop])

val_loss = model.evaluate(X_val, y_val)
y_pred_val = model.predict(X_val)
mape = np.mean(np.abs((y_val - y_pred_val.flatten()) / y_val)) * 100
print("Validation MAPE:", mape)

model.save('model.h5')
with open('scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

