
# LIBRARIES USED
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
import numpy as np
from datetime import date, datetime, timedelta
from numpy import ndarray
from streamlit_option_menu import option_menu
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
import seaborn as sns
from sklearn.metrics import r2_score
import feedparser
import ssl
import requests
ssl._create_default_https_context = ssl._create_unverified_context  # By-pass ssl certificate

# MAIN MENU SET-UP
st.set_page_config(page_title='predict crypto price', layout="wide")
from PIL import Image
logo = Image.open("Logo/pngegg.png")
col1, col2 = st.columns([0.3, 3])
with col1:
    st.image(logo, width=100)
with col2:
    st.markdown(
        """
        # SOLiGENCE Crypto Price Predictor
        """
    )
hide_menu_style = """
 <style>
 #MainMenu {visibility:hidden;}
 footer{visibility:hidden;}
 </style>
 """
st.markdown(hide_menu_style, unsafe_allow_html=True)
selected = option_menu(
    menu_title=None,
    options=["Home", "Forecast", "Indicator"],
    icons=["house", "clock-history", "bar-chart-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Period to load data from
start = '2011-01-01'
end = '2022-01-01'
today = date.today().strftime("%Y-%m-%d")

# HOME PAGE
if selected == "Home":
    cryptos = ("BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOT-USD",
               "BCH-USD", "LTC-USD", "LINK-USD", "UNI-USD", "DOGE-USD", "MATIC-USD",
               "FIL-USD", "EOS-USD", "BNB-USD", "XLM-USD", "TRX-USD", "XTZ-USD", "ATOM-USD", "VET-USD")
    targetcrypto = st.selectbox("Select a Crypto Currency", cryptos)

    # Loading our data function
    def load_data(ticker):
        data = yf.download(ticker, start, today)
        data.reset_index(inplace=True)
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data['Percent Change'] = (data['Close'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
        return data

    # Data loading status
    data_load_state = st.text("loading data...")
    data = load_data(targetcrypto)
    data_load_state.text("data loaded successfully....")
    st.subheader("Historical Data")

    def org_graph():
        # Add moving averages
        ma_list = [10, 20, 50]
        for ma in ma_list:
            data[f"MA{ma}"] = data['Close'].rolling(ma).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))

        # Add moving average traces to the graph
        for ma in ma_list:
            fig.add_trace(go.Scatter(x=data['Date'], y=data[f"MA{ma}"], name=f"MA{ma}"))

        fig.layout.update(title_text="TILL DATE DATA", xaxis_rangeslider_visible=True, height=600)
        st.plotly_chart(fig, use_container_width=True)
    org_graph()
    st.write('Double click on column titles to sort')
    st.write(data.tail(50))

    # TO BRING RSS TO THE HOMEPAGE
    def parse_feed(url):
        # Parse the RSS feed using the feedparser library
        news = feedparser.parse(url)
        # Create a list to hold the dictionaries for the 10 most recent entries in the RSS feed
        entries = []
        # Loop through the 10 most recent entries in the RSS feed and extract the relevant information
        for entry in news.entries[:10]:
            entry_dict = {}
            entry_dict['title'] = entry.title
            entry_dict['author'] = entry.author
            entry_dict['link'] = entry.link
            entry_dict['date'] = entry.published
            entry_dict['summary'] = entry.summary
            entries.append(entry_dict)
        return entries


    st.header("Trending")
    import streamlit as st

    url = 'https://cryptobriefing.com/feed/'
    entries = parse_feed(url)
    col1, col2 = st.columns([2, 3])

    # Display the most recent 10 news articles in the left column
    with col1:
        st.subheader('Breaking News')
        for entry in entries:
            st.write(entry['title'])

    # Display additional information for the selected article in the right column
    with col2:
        st.subheader('Selected Article')
        selected_entry = st.selectbox(
            'Select an article',
            [entry['title'] for entry in entries]
        )

        for entry in entries:
            if entry['title'] == selected_entry:
                st.write('Title:', entry['title'])
                st.write('Author:', entry['author'])
                st.write('Link:', entry['link'])
                st.write('Date:', entry['date'])
                st.write('Summary:', entry['summary'])

if selected == "Forecast":
    # DEFINING ALL FUNCTIONS NEEDED FOR TIMESERIES ANALYSIS
    @st.cache_data(show_spinner=False)
    def load_model(crypto):
        model = keras.models.load_model(f"Trained_Coins/{crypto}_lstm_model.h5")
        return model

    @st.cache_data(show_spinner=False)
    def load_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data

    def preprocess_data(data):
        close_data = data["Close"].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(close_data)
        return scaled_data, scaler

    def create_input_sequence(data, look_back):
        input_sequence = []
        for i in range(len(data) - look_back):
            input_sequence.append(data[i:i + look_back, 0])
        return np.array(input_sequence)

    def predict_prices(model, input_data):
        predicted_price = model.predict(input_data)
        return predicted_price


    def calculate_profit_threshold(crypto, duration, predicted_price, actual_price, profit_percent):
        # Get the index of the last day of the selected duration
        if duration == '1 Day':
            last_day_index = 0
        elif duration == '1 Week':
            last_day_index = 7
        elif duration == '1 Month':
            last_day_index = 30
        elif duration == '3 Months':
            last_day_index = 90
        else:
            last_day_index = 180

        # Get the last predicted price and the last historical price
        last_predicted_price = predicted_price[last_day_index]
        last_actual_price = predicted_price[-1]

        # Calculate the potential profit at the end of the duration
        potential_profit = (last_predicted_price - last_actual_price) / last_actual_price * 100

        # Print the values
        st.write("Potential profit/loss %:", potential_profit)
        st.write(f"Closing price in {duration}:", last_predicted_price)
        st.write("Closing price as at today:", last_actual_price)

        # Convert profit_percent to float
        profit_percent = float(profit_percent)

        # Code for checking if potential profit is greater than or equal to the desired profit percentage
        if potential_profit >= profit_percent and last_predicted_price > last_actual_price:
            return f"You can invest in {crypto} to achieve a profit of at least {profit_percent}% at the end of {duration}."
        else:
            return f"The profit threshold cannot be reached with {crypto} at the end of {duration}. It might result in a loss."


    # CREATING THE STREAMLIT APP FOR PREDICTING
    st.header('Make a Prediction')
    ticker = st.selectbox('Select Cryptocurrency', ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOT-USD",
                                                    "BCH-USD", "LTC-USD", "LINK-USD", "UNI-USD", "DOGE-USD",
                                                    "MATIC-USD",
                                                    "FIL-USD", "EOS-USD", "BNB-USD", "XLM-USD", "TRX-USD", "XTZ-USD",
                                                    "ATOM-USD", "VET-USD"])
    duration = st.selectbox('Select Duration to Predict',
        ['1 Day', '1 Week', '1 Month', '3 Months', '6 Months'])
    start_date = pd.to_datetime('2022-01-01')
    end_date = datetime.today()

    model = load_model(ticker)
    data = load_data(ticker, start_date, end_date)
    scaled_data, scaler = preprocess_data(data)

    if duration == '1 Day':
        look_back = 150
        num_days = 1
    elif duration == '1 Week':
        look_back = 150
        num_days = 7
    elif duration == '1 Month':
        look_back = 150
        num_days = 30
    elif duration == '3 Months':
        look_back = 150
        num_days = 90
    else:
        look_back = 150
        num_days = 180

    input_data: ndarray = create_input_sequence(scaled_data, look_back)
    input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
    predicted_price = predict_prices(model, input_data)
    predicted_price = scaler.inverse_transform(predicted_price)

    actual_price = data['Close'][look_back:]
    predicted_price = predicted_price.flatten()

    # DISPLAY PREDICTED PRICE FOR TODAY
    st.subheader("Today's Price")
    predicted_day_price = predicted_price[-1]
    st.write(f" Close price for {data['Date'].iloc[-1]}: {predicted_day_price:.2f}")

    # After calculating the predicted price
    actual_price = data['Close'][look_back:]
    st.write("Compare your desired profit to other coins by selecting another cryptocurrency")
    profit_percent = st.text_input("Enter your desired profit percentage %", value=0.00)
    try:
        number = float(profit_percent)  # Convert the input to a float
    except ValueError:
        st.error("Invalid input. Please enter a valid number.")
    else:
        # Perform calculations or desired actions with the valid number
        st.success(f"Entered number: {number}")
        profit_threshold_result = calculate_profit_threshold(ticker, duration, predicted_price, actual_price, profit_percent)

        st.subheader("Profit Threshold")
        st.write(profit_threshold_result)


    # PLOTTING OUR CHART WITH CANDLE STICKS
    future_dates = pd.date_range(end_date, end_date + timedelta(days=num_days), closed='right')
    fig = go.Figure()
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(x=data['Date'], open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'],
            name='Historical Data Chart'))
    # Add predicted prices trace
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_price.flatten(), name='Predicted Prices', mode='lines'))
    # Add a line connecting the last historical price to the first predicted price
    last_date = data['Date'].iloc[-1]
    fig.add_shape(type='line', x0=last_date, y0=data['Close'].iloc[-1], x1=future_dates[0], y1=predicted_price[0],
        line=dict(color='gray', width=1))
    fig.update_layout(title=f"{ticker} Price Prediction Chart ({duration})", xaxis_title='Date', yaxis_title='Price (USD)',
        height=600)
    st.plotly_chart(fig, use_container_width=True)

    # CONFIDENCE LEVEL
    st.subheader("Confidence Level")
    # MAE
    mae = np.mean(np.abs(predicted_price - actual_price))
    st.write(f"Mean Absolute Error: {mae:.2f}")
    # R2 Score
    r2 = r2_score(actual_price, predicted_price)
    st.write(f"R2-Score: {r2:.2f}")
    # MAPE
    diff = np.abs(predicted_price - actual_price)
    mean_abs_pct_error = np.mean(diff / actual_price) * 100
    st.write(f"Mean Absolute Percentage Error: {mean_abs_pct_error:.2f}%")

    # DISPLAY PREDICTED PRICE FOR TODAY
    st.header("Plan my purchase")
    predicted_day_price = predicted_price[-1]
    st.write(f"Close price for {data['Date'].iloc[-1]}: {predicted_day_price:.2f}")

    # CALCULATE CRYPTO AMOUNT FROM USD AMOUNT
    st.subheader('Sell my coin')
    st.write('Use the calculator to calculate profit/loss.')

    quantity = st.text_input('Enter the Quantity of Coins', value=1)
    price = st.text_input('What Price per coin? (Enter selling Price)', value=float(predicted_day_price))

    try:
        quantity = int(quantity)
        price = float(price)
        profit = (price - actual_price.iloc[-1]) * quantity
        st.write(
            f'If you sell {quantity} {ticker.split("-")[0]} coins at {price} USD per coin, you will make a profit of USD {profit:.2f}.')
    except ValueError:
        st.error('Invalid input. Please enter valid numbers for quantity and price.')

    st.subheader("Buy Coin")
    # CALCULATE CRYPTO AMOUNT FROM USD AMOUNT
    money = st.text_input("Enter the amount in USD", value=str(predicted_day_price))
    try:
        money = float(money)
    except ValueError:
        st.error("Invalid input. Please enter a valid number.")
    else:
        # Perform calculations or desired actions with the valid number
        st.success(f"Entered number: {money}")
        crypto_amount = money / float(actual_price.iloc[-1])
        st.write(f"With {money:.2f} USD, you can have {crypto_amount:.2f} {ticker.split('-')[0]} in your wallet.")

if selected == "Indicator":
    st.header('Correlation')
    # CORRELATED CURRENCIES AND HEATMAP
    import datetime as dt
    # List of available cryptocurrencies
    crypto = ['BTC', 'ETH', 'BCH', 'LTC', 'BNB', 'USDT', 'XRP', 'LINK', 'DOT',
              'ADA', 'BSV', 'CRO', 'EOS', 'XTZ', 'XLM', 'TRX', 'NEO', 'ATOM', 'VET', 'MKR']

    against_currency = "USD"

    start = dt.datetime(2022, 1, 1)
    end = dt.datetime.now()

    # Fetching historical price data for all cryptocurrencies
    data = pd.DataFrame()
    for crypto_currency in crypto:
        all_data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)
        all_data['crypto_currency'] = crypto_currency
        data = pd.concat([data, all_data])

    # Default cryptocurrency selection
    default_crypto = 'BTC'
    # Create a dropdown to select a cryptocurrency
    selected_crypto = st.selectbox('Select a cryptocurrency', crypto, index=crypto.index(default_crypto))
    # Filter the data to include only the selected cryptocurrency
    selected_data = data[data['crypto_currency'] == selected_crypto]
    # Calculate the correlation coefficients between the selected cryptocurrency and all other cryptocurrencies
    corr = data.groupby('crypto_currency').apply(lambda x: x['Close'].corr(selected_data['Close']))
    # Sort the correlation coefficients in descending order
    top_10_pos_corr = corr.sort_values(ascending=False).head(11)[1:]
    # Get the bottom 10 negative correlated cryptocurrencies
    bottom_10_neg_corr = corr[corr < 0].sort_values().head(10)
    # Calculate correlation matrix
    corr_matrix = data.pivot_table(index='Date', columns='crypto_currency', values='Close').corr()
    # Set up two columns for display
    col1, col2 = st.columns(2)

    # Display the results in a table
    with col1:
        st.write(f'Top 10 positive correlated crypto coin(s) to {selected_crypto}:')
        st.dataframe(top_10_pos_corr)

    with col2:
        if bottom_10_neg_corr.empty:
            st.write('No negative correlated crypto coin(s) found.')
        else:
            st.write(f'10 negative correlated crypto coin(s) to {selected_crypto}:')
            st.dataframe(bottom_10_neg_corr)



