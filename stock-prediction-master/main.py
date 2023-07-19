import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
import mplfinance as mpf

import streamlit as st
from datetime import date
import requests
from telegram import Bot

# @st.cache
def get_data():
    path = 'stock.csv'
    return pd.read_csv(path, low_memory=False)

df = get_data()
df = df.drop_duplicates(subset="Name", keep="first")

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")
st.write("###")

stocks = df['Name']
selected_stock = st.selectbox("Select dataset and years for prediction", stocks)

index = df[df["Name"]==selected_stock].index.values[0]
symbol = df["Symbol"][index]

n_years = st.slider("", 1, 5)
period = n_years * 365

# @st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data ...")
data = load_data(symbol)
data_load_state.text("Loading data ... Done!")

st.write("###")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Candlestick Chart
st.subheader("Candlestick Chart")
fig_candlestick = go.Figure(data=[go.Candlestick(x=data['Date'],
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'])])
st.plotly_chart(fig_candlestick)

# Moving Averages
st.subheader("Moving Averages")
ma_period = st.slider("Select Moving Average Period", 5, 50, 20)
data['MA'] = data['Close'].rolling(window=ma_period).mean()
fig_moving_averages = go.Figure()
fig_moving_averages.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Price'))
fig_moving_averages.add_trace(go.Scatter(x=data['Date'], y=data['MA'], name=f'{ma_period}-day Moving Average'))
fig_moving_averages.update_layout(title_text="Moving Averages", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_moving_averages)

# Bollinger Bands
st.subheader("Bollinger Bands")
bollinger_period = st.slider("Select Bollinger Bands Period", 10, 50, 20)
data['MA'] = data['Close'].rolling(window=bollinger_period).mean()
data['STD'] = data['Close'].rolling(window=bollinger_period).std()
data['Upper'] = data['MA'] + 2 * data['STD']
data['Lower'] = data['MA'] - 2 * data['STD']
fig_bollinger = go.Figure()
fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Price'))
fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Upper'], name='Upper Band'))
fig_bollinger.add_trace(go.Scatter(x=data['Date'], y=data['Lower'], name='Lower Band'))
fig_bollinger.update_layout(title_text="Bollinger Bands", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_bollinger)

# Volume Chart
st.subheader("Volume Chart")
fig_volume = go.Figure()
fig_volume.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
fig_volume.update_layout(title_text="Volume Chart", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_volume)

# Relative Strength Index (RSI)
st.subheader("Relative Strength Index (RSI)")
rsi_period = st.slider("Select RSI Period", 5, 50, 14)

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=rsi_period).mean()
avg_loss = loss.rolling(window=rsi_period).mean()

rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data['Date'], y=rsi, name='RSI'))
fig_rsi.update_layout(title_text="Relative Strength Index (RSI)", xaxis_rangeslider_visible=True)
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
st.plotly_chart(fig_rsi)

# MACD (Moving Average Convergence Divergence)
st.subheader("MACD (Moving Average Convergence Divergence)")

exp12 = data['Close'].ewm(span=12, adjust=False).mean()
exp26 = data['Close'].ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=data['Date'], y=macd, name='MACD'))
fig_macd.add_trace(go.Scatter(x=data['Date'], y=signal, name='Signal'))
fig_macd.add_trace(go.Bar(x=data['Date'], y=histogram, name='Histogram'))
fig_macd.update_layout(title_text="MACD (Moving Average Convergence Divergence)", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_macd)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.index, colorscale='Viridis'))
st.plotly_chart(fig_heatmap)


# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write("***")
st.write("###")

st.subheader("Forecast data")
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# # Candlestick chart for forecasted data
# fig3 = go.Figure(data=[go.Candlestick(x=forecast['ds'],
#                                      open=forecast['yhat_lower'],
#                                      high=forecast['yhat_upper'],
#                                      low=forecast['yhat_lower'],
#                                      close=forecast['yhat'],
#                                      increasing_line_color='green',
#                                      decreasing_line_color='red')])
# fig3.update_layout(title_text="Forecasted Data (Candlestick Chart)")
# st.plotly_chart(fig3)

# Send forecasted data to Telegram


# import asyncio
# from aiogram import Bot, types, Dispatcher
# from aiogram.types import Message

# async def send_telegram_message(bot_token, chat_id, message):
#     bot = Bot(token=bot_token)
#     await bot.send_message(chat_id=chat_id, text=message)

# async def send_forecast_to_telegram(forecast):
#     bot_token = '6326049405:AAFl-O0U9kaQzahf-LZXyhdJq57XdqO8JeU'
#     chat_id = '486276870'

#     message = f"Forecasted data:\n{forecast.tail().to_string()}"

#     await send_telegram_message(bot_token, chat_id, message)

# # Handler for the button click
# async def button_click_handler(message: Message):
#     forecast = ...  # Get the forecasted data here
#     await send_forecast_to_telegram(forecast)



