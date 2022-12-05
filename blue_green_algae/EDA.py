import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

pd.set_option('display.max_columns', 1000)


# data.set_index('time', inplace=True)

# plot
def get_plt(data, outlier, filter):
    data = data.drop('time', axis=1)
    if outlier:
        data = delete_outliers(data, filter)
    num_col = len(data.columns)
    plt.figure(figsize=(10, 20))
    for i, column in enumerate(data.columns):
        if column != 'time':
            plt.subplot(num_col, 1, i + 1)
            plt.plot(data[column])
            plt.title(column, loc='right', y=0.5)
    plt.show()
    return data


# descriptive stat
def get_desc_stat(data, outlier, filter):
    data = data.drop('time', axis=1)
    if outlier:
        data = delete_outliers(data, filter)

    data_describe = data.describe()
    index = data_describe.index[1:]
    plt.figure(figsize=(15, 10))
    for i in range(len(data.columns)):
        ax = plt.subplot(3, 3, i + 1)
        ax.set_title(data.columns[i])
        for j in range(len(index)):
            plt.bar(index[j], data_describe.loc[index[j], data.columns[i]])
    # plt.show()
    plt.savefig('/home/weiyichen/桌面/EDA_shape-dushu_20210901.png')
    return data


def delete_outliers(data, filter):
    if filter:
        for attr in data.columns:
            data[attr] = pd.DataFrame(savgol_filter(data[attr], window_length=11, polyorder=3))
    # delete minus values
    data = data[data > 0].dropna()
    data.reset_index(drop=True, inplace=True)

    data_norm = (data - data.mean()) / (data.std())
    data = data[abs(data_norm[:]) <= 3].dropna().reset_index(drop=True)
    plt.savefig('/home/weiyichen/桌面/EDA_dist-dushu_20210901.png')
    return data


attrs = ['时间', '叶绿素', '电导率', '溶解氧(mg/L)', '藻蛋白', '总溶解固体', '浊度', '温度', 'PH值']
data = pd.read_csv('./datasets/监测数据1/监测数据-独墅-20210901.csv')[attrs]

data.rename(columns={'时间': 'time', '叶绿素': 'Chlo', '电导率': 'Cond',
                     '溶解氧(mg/L)': 'DO', '藻蛋白': 'Phyco',
                     '总溶解固体': 'TDS', '浊度': 'Turb',
                     '温度': 'Temp', 'PH值': 'pH'}, inplace=True)
data['time'] = pd.to_datetime(data['time'])
data['Phyco'] /= 100
data = data.sort_values('time')
data.reset_index(drop=True, inplace=True)

# DELETE outliers
data_processed = get_plt(data, True, False)
get_desc_stat(data, True, False)

data_unprocessed = get_plt(data, False, False)
get_desc_stat(data, False, False)
