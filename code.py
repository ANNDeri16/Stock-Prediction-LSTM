# %% [markdown]
# ## Project: Stock Price Prediction [LSTM Using 60 Day Stock Price]

# %% [markdown]
# **Step 1**: Library

# %%
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# %% [markdown]
# **Step 2**: Get the data of desired firm from website. 
# 

# %%
df = web.DataReader("BRIS.JK", data_source="yahoo", start='2020-01-01', end='2022-04-20')
df

# %%
#Number of rows and colomns in the dataset
df.shape

# %%
#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Closing Price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD (IDR)',fontsize=18)
plt.show()


# %% [markdown]
# **Step 3**: Data Prep

# %%
#Create a new dataframe with only closing price
data = df.filter(['Close'])
#Convert to a numpy array
dataset = data.values
#Indexing for train data
traning_data_len = math.ceil(len(dataset)*.8)
traning_data_len

# %%
#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data
len(scaled_data)

# %%
#Create the training dataset
train_data = scaled_data[0:traning_data_len,:]
#split data into x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

# %%
#Convert the x dan y train to numpy
x_train, y_train = np.array(x_train), np.array(y_train)

# %%
#Reshape data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

# %% [markdown]
# **Step 4**: Modeling 

# %%
#Built LSTM Model
Model = Sequential()
Model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
Model.add(LSTM(50, return_sequences=False))
Model.add(Dense(25))
Model.add(Dense(1))

# %%
#Compile the model
Model.compile(optimizer='adam', loss='mean_squared_error')

# %%
#Train the model
Model.fit(x_train, y_train, batch_size=1, epochs=1)

# %% [markdown]
# **Step 5**: Evaluation 

# %%
#Create testing dataset
test_data = scaled_data[traning_data_len-60:,:]
x_test = []
y_test = dataset[traning_data_len:,:]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i,0])
    

# %%
x_test = np.array(x_test)

# %%
 x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

# %%
#Get the models predicted price values
predictions = Model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# %%
#Evaluate teh model (RMSE)
rmse = np.sqrt(np.mean((predictions-y_test)**2))
rmse

# %%
#Plot
train = data[:traning_data_len]
valid = data[traning_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price IDR',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# %%
valid

# %%
#Get the quote
bris_quote = web.DataReader("BRIS.JK", data_source="yahoo", start='2020-01-01', end='2022-04-20')
#Create a new dataframe
new_df = bris_quote.filter(['Close'])
#Get the last 60 closing price values and convert to array
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_price = Model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

# %%
bris_quote2 = web.DataReader("BRIS.JK", data_source="yahoo", start='2020-01-01', end='2022-04-21')
print(bris_quote2['Close'])

