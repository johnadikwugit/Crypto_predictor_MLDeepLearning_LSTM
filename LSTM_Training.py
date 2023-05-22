# Libraries and packages
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def load_data(ticker): # Function for loading our dataset
    end_date = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start='2021-01-01', end=end_date)
    data.reset_index(inplace=True)
    return data


def create_model(look_back): # Architecture
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Creating dataset for model
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Scale the data
scaler = StandardScaler()

# Load data for selected cryptocurrencies and train models
targetcrypto = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOT-USD",
                "BCH-USD", "LTC-USD", "LINK-USD", "UNI-USD", "DOGE-USD", "MATIC-USD",
                "FIL-USD", "EOS-USD", "BNB-USD", "XLM-USD", "TRX-USD", "XTZ-USD", "ATOM-USD", "VET-USD"]

for crypto in targetcrypto:
    # Load data for cryptocurrency
    data = load_data(crypto)

    # We dropped missing data
    data = data.dropna()

    # Convert data to numpy array
    close_data = data["Close"].values.reshape(-1, 1)

    # Scale the data
    scaled_data = scaler.fit_transform(close_data)

    # Splitted data into training and testing sets
    training_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[0:training_size, :]
    test_data = scaled_data[training_size:len(scaled_data), :]

    # Created training and testing datasets for LSTM model
    look_back = 150
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    # Reshaped input data for LSTM model
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # Training LSTM model
    model = create_model(look_back)
    model.fit(trainX, trainY, epochs=50, batch_size=32)

    loss = model.evaluate(testX, testY)
    print('Final loss:', loss)

    # Saving the trained model
    model.save(f"Trained_Coins/{crypto}_lstm_model.h5")
