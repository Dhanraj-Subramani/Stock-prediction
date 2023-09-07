import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
# For time stamps
from datetime import datetime
en = datetime.now()
st = datetime(en.year - 10, en.month, en.day)
df = pdr.get_data_yahoo('BABA', start=st, end=en)
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil( len(dataset) ))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
from keras.models import Sequential
from keras.layers import Dense, LSTM,Bidirectional,Dropout,Flatten

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),])
model.fit(x_train, y_train, batch_size=1, epochs=1)
model.save('./Models/sub.h5')
print("Model saved successfully")