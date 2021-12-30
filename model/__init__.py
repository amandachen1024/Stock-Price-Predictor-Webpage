import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import matplotlib.dates as mdates
import datetime as dt
import yfinance as yf
from numpy import floor
from sklearn.metrics import accuracy_score
yf.pdr_override()

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense # LSTM - long short term memory

# Load Data
def predict(ticker, start_price, num_shares, future_days):
    try:
        company = yf.Ticker(ticker)
        start = dt.datetime(2012, 1, 1)
        end = dt.datetime(2020, 1, 1)
        data = company.history(start=start, end=end)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(
        data["Close"].values.reshape(-1, 1))  # only predicting closing prices, reshape reshapes the array
    except:
        return -1

    prediction_days = 100

    # prepare training data
    x_train = []
    y_train = []

    # add 60 values and then 61st value as training example
        # model has 60 values and then tries to predict the next value and can check against 61st value
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0]) # first 60 values, 0 signifies column (?)
        y_train.append(scaled_data[i, 0]) # 61st value

    # convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape x_train to fit with neural network
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) # add additional dimension

    # Build model
    model = Sequential() # basic neural network

    # Add layers
    # add pattern of LSTM then Dropout, and then add 1 Dense layer which will be predictions
    # units: number of layers, LSTM is recurrent so it feeds back information so return_sequences is true (instead of just forwarding like Dense layer)
    num_units = 50
    model.add(LSTM(units=num_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=num_units, return_sequences=True)) # already know input shape
    model.add(Dropout(0.2))
    model.add(LSTM(units=num_units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # prediction of next closing value

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # epochs: model sees same data 24 times, batch_size: model sees 32 units at once
    # can also save and load model

    # Test Model
    # see how well it predicts already known data - must be data model has not seen

    # load test data after what model has seen
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = company.history(start=test_start, end=test_end)
    actual_prices = test_data["Close"].values # must save correct data

    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0) # combine training and testing data

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    # what model sees, try to start as early as possible
    # reshape and scale input
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data

    x_test = []

    for i in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[i - prediction_days: i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # print(x_test.shape)
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # predicting based on code found at https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816
    total_dataset = total_dataset.values.reshape((-1))
    prediction_list = model_inputs[-prediction_days:]
    prediction_list = np.reshape(prediction_list, (-1, 1))
    # prediction_list = scaler.transform(prediction_list)
    prediction_list = np.reshape(prediction_list, (-1))

    for _ in range(future_days):
        x = prediction_list[-prediction_days:]
        x = np.reshape(x, (1, prediction_days, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)

    prediction_list = prediction_list[prediction_days:]

    prediction = np.reshape(prediction_list[-1], (1, -1))
    prediction = scaler.inverse_transform(prediction)
    prediction_amount = prediction[0][0]

    last_date = test_end
    prediction_dates = pd.date_range(last_date, periods=future_days).tolist()
    prediction_plot = np.reshape(prediction_list, (-1, 1))
    prediction_plot = scaler.inverse_transform(prediction_plot)

    # Visualize / Plot Test Predictions
    '''test_data["Date"]=pd.to_datetime(test_data.Date,format="%Y-%m-%d")
    test_data.index=test_data['Date']'''
    plt.clf()
    plt.plot(test_data.index, actual_prices, color="black", label=f"Actual {ticker} price")
    plt.plot(test_data.index, predicted_prices, color="green", label=f"Predicted {ticker} price")
    plt.plot(prediction_dates, prediction_plot, color="blue", label=f"Future predicted {ticker} price")
    plt.xticks(rotation=45)
    plt.title(f"{ticker} Share Price")
    plt.xlabel("Date")
    plt.ylabel(f"{ticker} Share Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig('webpage/static/plot.png')

    per_share = prediction_amount
    net_per_share = prediction_amount - start_price
    total = prediction_amount * num_shares
    net_total = (prediction_amount - start_price) *num_shares
    print(f"Predicted Price(per share): ${per_share:.2f}")
    print(f"Net (per share): ${net_per_share:.2f}")
    print(f"Predicted Total Value: ${total:.2f}")
    print(f"Net Total: ${net_total:.2f}")

    return [per_share, net_per_share, total, net_total]



'''
future_prices = predicted_prices
mod = len(future_prices) % prediction_days
quotient = int(floor(len(future_prices) / prediction_days))
future_prices = np.reshape(future_prices[mod:], (quotient,prediction_days, 1))
#print(future_prices)


#print(model.predict(future_prices))
# predicted prices will be scaled so need to unscale them

for i in range(future_days):
    prediction = (model.predict(future_prices))[-1][0]
    print(scaler.inverse_transform(prediction.reshape(1, -1)))
    future_prices = np.append(future_prices, prediction)
    mod = len(future_prices) % prediction_days
    quotient = int(floor(len(future_prices) / prediction_days))
    future_prices = np.reshape(future_prices[mod:], (quotient,prediction_days, 1))


future_prices = np.reshape(future_prices, (quotient * prediction_days, 1))
future_prices = scaler.inverse_transform(future_prices)


predicted_prices = scaler.inverse_transform(predicted_prices)
score = model.evaluate(predicted_prices, actual_prices, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
prediction_amount = future_prices[-1][-1]
print(f"Predicted Price(per share): ${prediction_amount:.2f}")
print(f"Net (per share): ${prediction_amount-start_price:.2f}")
print(f"Predicted Total Value: ${prediction_amount * num_shares:.2f}")
print(f"Net Total: ${(prediction_amount-start_price) * num_shares:.2f}")
'''
# Predict Next Day
'''

while True:
    try:
        future_days = int(input("How many days into the future would you like to predict? "))
        real_data = np.array([model_inputs[len(model_inputs) + future_days - prediction_days:, 0]])
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        prediction_amount = prediction[0][0]
        print(f"Predicted Price(per share): ${prediction_amount:.2f}")
        print(f"Net (per share): ${prediction_amount-start_price:.2f}")
        print(f"Predicted Total Value: ${prediction_amount * num_shares:.2f}")
        print(f"Net Total: ${(prediction_amount-start_price) * num_shares:.2f}")
        break
    except:
        print("Invalid number of days.")
'''