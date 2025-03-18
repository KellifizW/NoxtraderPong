import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 特徵工程函數
def feature_engineering(data):
    """根據論文生成特徵"""
    data['Returns'] = data['Close'].pct_change()  # 日回報率
    data['ReturnMomentum'] = data['Returns'].diff()  # 回報動量
    data['ReturnAcceleration'] = data['ReturnMomentum'].diff()  # 回報加速度
    data['WeekPriceMomentum'] = data['Close'].pct_change(periods=5)  # 周價格動量
    data['MonthPriceMomentum'] = data['Close'].pct_change(periods=20)  # 月價格動量
    data['VolumeVelocity'] = data['Volume'].pct_change()  # 交易量速度
    data = data.dropna()  # 移除 NaN 值
    return data

# 訓練與預測函數
def train_and_predict(data, timesteps=10, epochs=50):
    """每 10 天訓練一次 LSTM 模型並預測 Return Momentum"""
    features = ['Returns', 'ReturnMomentum', 'ReturnAcceleration', 'WeekPriceMomentum', 'MonthPriceMomentum', 'VolumeVelocity']
    target = 'ReturnMomentum'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    scaled_features = scaler_features.fit_transform(data[features])
    scaled_target = scaler_target.fit_transform(data[[target]])

    predictions = []
    prediction_dates = []
    
    # 每 10 天一個區間進行訓練和預測
    for i in range(0, len(data) - timesteps, timesteps):
        X_train = []
        y_train = []
        # 構建訓練數據
        for j in range(i, min(i + timesteps, len(data) - timesteps)):
            X_train.append(scaled_features[j:j + timesteps])
            y_train.append(scaled_target[j + timesteps])

        if len(X_train) == 0:
            break

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 定義簡單的 LSTM 模型
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(timesteps, len(features))))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # 訓練模型
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        # 預測下一個 10 天
        X_test = scaled_features[i + timesteps:i + 2 * timesteps]
        if len(X_test) < timesteps:
            break
        X_test = X_test.reshape((1, timesteps, len(features)))
        pred = model.predict(X_test, verbose=0)
        pred = scaler_target.inverse_transform(pred)
        predictions.extend(pred.flatten())
        prediction_dates.extend(data.index[i + timesteps:i + timesteps + len(pred)])

    return predictions, prediction_dates

# 過濾股票
def filter_stocks(predictions, threshold):
    """根據閾值過濾預測值，小於閾值的設為無交易"""
    filtered_predictions = []
    for pred in predictions:
        if abs(pred) > threshold:
            filtered_predictions.append(pred)
        else:
            filtered_predictions.append(0)  # 標記為無交易
    return filtered_predictions

# 資本分配
def allocate_capital(filtered_predictions, capital):
    """對非零預測值進行等權重資本分配"""
    non_zero_preds = [pred for pred in filtered_predictions if pred != 0]
    if not non_zero_preds:
        return [0] * len(filtered_predictions)
    weight = 1 / len(non_zero_preds)
    positions = [weight * capital if pred != 0 else 0 for pred in filtered_predictions]
    return positions

# 回測函數
def backtest(data, predictions, prediction_dates, threshold, initial_capital=100000):
    """模擬交易並計算回測結果"""
    filtered_predictions = filter_stocks(predictions, threshold)
    positions = allocate_capital(filtered_predictions, initial_capital)

    portfolio_value = initial_capital
    portfolio_values = [initial_capital]
    daily_returns = []

    # 確保日期對齊
    pred_df = pd.DataFrame({'Prediction': filtered_predictions, 'Position': positions}, index=prediction_dates)
    data = data.join(pred_df).dropna()

    for i in range(1, len(data)):
        daily_return = data['Returns'].iloc[i]
        position_value = positions[i - 1] * (1 + daily_return)
        portfolio_value += position_value - positions[i - 1]
        portfolio_values.append(portfolio_value)
        daily_returns.append(daily_return)

    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    annualized_return = ((1 + total_return / 100) ** (252 / len(daily_returns))) - 1 * 100
    max_drawdown = max([1 - v / max(portfolio_values[:i + 1]) for i, v in enumerate(portfolio_values)]) * 100

    return portfolio_values, total_return, annualized_return, max_drawdown, data.index

# 主程式
def main():
    st.title("Nox Trader 策略回測系統")

    # 用戶輸入
    stock_symbol = st.text_input("輸入股票代碼（例如：AAPL）", value="AAPL")
    start_date = st.date_input("回測開始日期", value=datetime(2020, 1, 1))
    end_date = st.date_input("回測結束日期", value=datetime(2023, 1, 1))
    threshold = st.slider("過濾閾值（Return Momentum 幅度）", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    epochs = st.slider("訓練次數（epochs）", min_value=10, max_value=100, value=50, step=10)
    initial_capital = st.number_input("初始資金（美元）", min_value=1000, value=100000, step=1000)

    if st.button("運行回測"):
        with st.spinner("正在下載數據並運行回測..."):
            # 下載數據
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                st.error("無法獲取數據，請檢查股票代碼或日期範圍！")
                return

            # 特徵工程
            data = feature_engineering(data)

            # 訓練與預測
            predictions, prediction_dates = train_and_predict(data, epochs=epochs)

            # 回測
            portfolio_values, total_return, annualized_return, max_drawdown, dates = backtest(
                data, predictions, prediction_dates, threshold, initial_capital
            )

            # 顯示結果
            st.subheader("投資組合累積回報")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=portfolio_values, mode='lines', name='Portfolio Value'))
            fig.update_layout(
                title=f"{stock_symbol} 累積回報曲線",
                xaxis_title="日期",
                yaxis_title="投資組合價值（美元）",
                height=600,
                width=1000
            )
            st.plotly_chart(fig)

            st.subheader("性能指標")
            st.write(f"總回報率: {total_return:.2f}%")
            st.write(f"年化回報率: {annualized_return:.2f}%")
            st.write(f"最大回撤: {max_drawdown:.2f}%")
            st.write(f"初始資金: ${initial_capital}")
            st.write(f"最終資金: ${portfolio_values[-1]:.2f}")

if __name__ == "__main__":
    main()
