import matplotlib.pyplot as plt
import pandas as pd


# data.set_index('time', inplace=True)

# plot
def get_plt(data):
    num_col = len(data.columns)
    plt.figure(figsize=(10, 20))
    for i, column in enumerate(data.columns):
        ax = plt.subplot(num_col, 1, i + 1)
        plt.plot(data[column])
        ax.set_title(column)
    plt.show()


# descriptive stat
def get_desc_stat(data):
    data = data.drop('time', axis=1)
    data_describe = data.describe()
    index = data_describe.index[1:]
    plt.figure(figsize=(15, 10))
    for i in range(len(data.columns)):
        ax = plt.subplot(3, 3, i + 1)
        ax.set_title(data.columns[i])
        for j in range(len(index)):
            plt.bar(index[j], data_describe.loc[index[j], data.columns[i]])
    plt.show()


attrs = ['时间', '叶绿素', '电导率', '溶解氧(mg/L)', '藻蛋白', '总溶解固体', '浊度', '温度', 'PH值']
data = pd.read_csv('./datasets/监测数据1/监测数据-独墅-20210901.csv')[attrs]

data.rename(columns={'时间': 'time', '叶绿素': 'Chlo', '电导率': 'Cond',
                     '溶解氧(mg/L)': 'DO', '藻蛋白': 'Phyco',
                     '总溶解固体': 'TDS', '浊度': 'Turb',
                     '温度': 'Temp', 'PH值': 'pH'}, inplace=True)
data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')
data.reset_index(drop=True, inplace=True)

get_plt(data)
get_desc_stat(data)

