from flask import Flask, render_template, request
import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import holidays
from datetime import datetime, timedelta

import db


app = Flask(__name__)

model = load_model('model.h5')
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

bd_holidays = holidays.BD()

def calculate_accuracy(actual_price, predicted_price):
    try:
        accuracy = (1 - abs(predicted_price - actual_price) / actual_price) * 100
        return accuracy
    except ZeroDivisionError:
        return None

def get_next_business_days(start_date, n=7):
    days = []
    current = start_date + timedelta(days=1)
    while len(days) < n:
        if current.weekday() not in (4, 5) and current not in bd_holidays:
            days.append(current)
        current += timedelta(days=1)
    return days

@app.route('/', methods=['GET', 'POST'])
def index():
    conn = sqlite3.connect('dse.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT [TRADING CODE] FROM dse_data")
    companies = [row[0] for row in c.fetchall()]
    conn.close()

    prediction = last_close = accuracy = None
    next_dates = next_preds = []

    if request.method == 'POST':
        trading_code = request.form.get('trading_code')
        conn = sqlite3.connect('dse.db')
        c = conn.cursor()
        c.execute("""
            SELECT DATE, YCP
            FROM dse_data
            WHERE [TRADING CODE] = ?
            ORDER BY DATE DESC
            LIMIT 1
        """, (trading_code,))
        row = c.fetchone()
        conn.close()

        if row:
            last_date_str, last_close = row
            last_close = float(last_close)
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')

            df = pd.read_csv('dse_data.csv', thousands=',')
            df = df[df['TRADING CODE'] == trading_code].sort_values('DATE')
            window_size = 10

            if len(df) >= window_size:
                recent = df['CLOSEP*'].values[-window_size:].astype('float32')
                scaler = scalers.get(trading_code)
                if scaler:
                    window = scaler.transform(recent.reshape(-1, 1)).flatten()
                    X_input = window.reshape(1, window_size, 1)
                    pred_scaled = model.predict(X_input)
                    prediction = float(scaler.inverse_transform(pred_scaled)[0][0])
                    prediction = round(prediction, 2)
                    accuracy = calculate_accuracy(last_close, prediction)

                    next_days = get_next_business_days(last_date, n=7)
                    preds = []
                    rolling = window.copy()
                    for _ in next_days:
                        Xn = rolling.reshape(1, window_size, 1)
                        ps = model.predict(Xn)
                        val = float(scaler.inverse_transform(ps)[0][0])
                        preds.append(round(val, 2))
                        new_scaled = scaler.transform(np.array([[val]]))[0, 0]
                        rolling = np.roll(rolling, -1)
                        rolling[-1] = new_scaled

                    next_dates = [d.strftime('%a, %b %d') for d in next_days]
                    next_preds = preds
                else:
                    prediction = "Scaler not found"
            else:
                prediction = "Insufficient historical data"

    return render_template(
        'index.html',
        companies=companies,
        prediction=prediction,
        last_close=last_close,
        accuracy=accuracy,
        next_dates=next_dates,
        next_preds=next_preds
    )

if __name__ == '__main__':
    app.run(debug=True)
