# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

# import pandas_profiling as pp


# def data_report(data):
#     report_total = pp.ProfileReport(data)
#     report_total.to_file('./report_total.html')


data = pd.read_csv('test1.csv')
# data_report(data)
data = data.dropna()
data['date'] = pd.to_datetime(data['date'])
# Preserve 5 items
top_5 = data.groupby(['item'])['date'].count().sort_values(ascending=False)


def pre_processing(data, name: str):
    tmp = data.loc[data['item'] == name].drop('item', axis=1) \
        .groupby('date') \
        .agg({'sale_cnt': 'sum', 'price': 'mean'})
    tmp = tmp[(tmp != 0).all(1)]
    return tmp


def plot(data, name: str):
    fg, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax[0].plot(data['sale_cnt'], label='sale_cnt', color='green')
    ax[0].set_xlabel('date', size=15)
    ax[0].set_ylabel('sale_cnt', size=15)
    ax[0].legend()
    plt.title(name)

    ax[1].plot(data['price'], label='price', color='red')
    ax[1].set_xlabel('date', size=15)
    ax[1].set_ylabel('price', size=15)
    ax[1].legend()
    plt.title(name)
    fg.show()


def create_sequence(dataset):
    sequences = []
    labels = []

    start_idx = 0

    for stop_idx in range(50, len(dataset)):  # Selecting 50 rows at a time
        sequences.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return np.array(sequences), np.array(labels)


def visualization_actual_pred(gs_slic_data, name: str):
    gs_slic_data[['sale_cnt', 'price']] = MMS.inverse_transform(gs_slic_data[['sale_cnt', 'price']])  # Inverse scaling
    gs_slic_data[['sale_cnt', 'sale_cnt_predicted']].plot(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel(name, size=15)
    plt.title('Actual vs Predicted for sale_cnt', size=15)
    plt.show()

    gs_slic_data[['price', 'price_predicted']].plot(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel(name, size=15)
    plt.title('Actual vs Predicted for price', size=15)
    plt.show()


def visualization_upcoming(gs_slic_data, upcoming_prediction):
    fg, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gs_slic_data.loc[:'2022-06-11', 'sale_cnt'], label='Current sale_cnt')
    ax.plot(upcoming_prediction.loc['2022-06-11':, 'sale_cnt'], label='Upcoming sale_cnt')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlabel('Date', size=15)
    ax.set_ylabel('80%VB2', size=15)
    ax.set_title('Upcoming sale_cnt prediction', size=15)
    ax.legend()
    fg.show()

    fg, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gs_slic_data.loc[:'2022-06-11', 'price'], label='Current price')
    ax.plot(upcoming_prediction.loc['2022-06-11':, 'price'], label='Upcoming price')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    ax.set_xlabel('Date', size=15)
    ax.set_ylabel('80%VB2', size=15)
    ax.set_title('Upcoming price prediction', size=15)
    ax.legend()
    fg.show()


VB2_80 = pre_processing(data, '80_VB2')
VB6 = pre_processing(data, 'VB6')
VH_02 = pre_processing(data, '2_VH')
VB2_98 = pre_processing(data, '98_VB2')
Griseofulvin = pre_processing(data, '灰黄霉素')

# plot(VB2_80, '80_VB2')
# Scale the data
MMS = MinMaxScaler()
VB2_80[VB2_80.columns] = MMS.fit_transform(VB2_80)
training_size = round(len(VB2_80) * 0.80)  # Selecting 80 % for training and 20 % for testing
train_data = VB2_80[:training_size]
test_data = VB2_80[training_size:]

# Create seq
train_seq, train_label = create_sequence(train_data)
test_seq, test_label = create_sequence(test_data)

# Create LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
# model.summary()

model.fit(train_seq, train_label, epochs=100, validation_data=(test_seq, test_label), verbose=1)
test_predicted = model.predict(test_seq)
test_inverse_predicted = MMS.inverse_transform(test_predicted)  # Inversing scaling on predicted data

# Visualization
gs_slic_data = pd.concat([VB2_80.iloc[-151:].copy(),
                          pd.DataFrame(test_inverse_predicted, columns=['sale_cnt_predicted', 'price_predicted'],
                                       index=VB2_80.iloc[-151:].index)], axis=1)
visualization_actual_pred(gs_slic_data, 'VB2_80')

# Forecasting for the upcoming 10 days
# Creating a dataframe and adding 10 days to existing index
gs_slic_data = gs_slic_data.append(pd.DataFrame(columns=gs_slic_data.columns,
                                                index=pd.date_range(start=gs_slic_data.index[-1], periods=11, freq='D',
                                                                    closed='right')))
upcoming_prediction = pd.DataFrame(columns=['sale_cnt', 'price'], index=pd.DatetimeIndex(gs_slic_data.index))
upcoming_prediction.index = pd.to_datetime(upcoming_prediction.index)
curr_seq = test_seq[-1:]

for i in range(-10, 0):
    up_pred = model.predict(curr_seq)
    upcoming_prediction.iloc[i] = up_pred
    curr_seq = np.append(curr_seq[0][1:], up_pred, axis=0)
    curr_seq = curr_seq.reshape(test_seq[-1:].shape)
upcoming_prediction[['sale_cnt', 'price']] = MMS.inverse_transform(upcoming_prediction[['sale_cnt', 'price']])

visualization_upcoming(gs_slic_data, upcoming_prediction)
