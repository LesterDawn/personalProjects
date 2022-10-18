import os
import io
import math
import pandas as pd
# import pandas_profiling as pp
from scipy.stats import kstest
from scipy.stats import normaltest
from sklearn.decomposition import PCA
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def load_data(directory: str):
    df = pd.DataFrame()
    # drop_list = ['salinity', 'ammonia nitrogen', 'longitude', 'latitude', 'dissolved oxygen (%Sat)', 'PH value (mv)']
    drop_list = ['salinity', 'ammonia nitrogen', 'dissolved oxygen (%Sat)', 'PH value (mv)']
    for file in os.listdir(directory):
        if file != 'raw_data':
            df = df.append(pd.read_csv(directory + file, index_col=None).drop(drop_list, axis=1))
    df['time'] = pd.to_datetime(df['time']).dt.date
    return df.reset_index().drop('index', axis=1)


def pca_data(data):
    # Normalization
    data_scaled = pd.DataFrame(preprocessing.scale(data), columns=data.columns)
    # PCA reduction
    pca = PCA(n_components='mle')
    data_reduced = pca.fit_transform(data_scaled)
    pca_info = pd.DataFrame(pca.components_, columns=data_scaled.columns)
    pca_var_ratio = pca.explained_variance_ratio_

    return data_reduced, pca_info, pca_var_ratio


# def data_report(data):
#     report_total = pp.ProfileReport(data)
#     report_total.to_file('./report_total.html')


def correlation(data):  # get corr of features
    temp = preprocessing.scale(data)
    data_scaled = pd.DataFrame(preprocessing.scale(data), columns=data.columns)
    return data_scaled.corr(method="pearson")


def concat_csv(f1, f2, directory):
    file1 = pd.read_csv(directory + f1, index_col=None)
    file2 = pd.read_csv(directory + f2, index_col=None)
    file = [file1, file2]
    # drop_list = ['salinity', 'ammonia nitrogen', 'dissolved oxygen (%Sat)', 'PH value (mv)']
    train = pd.concat(file)  # .drop(drop_list, axis=1)
    if f1.__contains__('独墅湖'):
        print(directory + "监测数据-独墅-" + f1.split('-')[2])
        train.to_csv(directory + "监测数据-独墅-" + f1.split('-')[2], index=False, encoding='utf_8_sig')
    elif f1.__contains__('金鸡湖'):
        print(directory + "监测数据-金鸡-" + f1.split('-')[2])
        train.to_csv(directory + "监测数据-金鸡-" + f1.split('-')[2], index=False, encoding='utf_8_sig')


def merge_data(directory: str):
    # raw data: ./datasets/jinji_lake/raw_data
    # merged data: ./dataset/jinji_lake
    inside = []
    outside = []
    for file in os.listdir(directory):
        if file.__contains__('内湖'):
            inside.append(file)
        elif file.__contains__('外湖'):
            outside.append(file)
    for i in inside:
        for o in outside:
            if o.split('-')[2] == i.split('-')[2]:
                inside.remove(i)
                outside.remove(o)
                concat_csv(i, o, directory)
                break
    for i in inside:
        df = pd.read_csv(directory + i, index_col=None)
        if i.__contains__('独墅湖'):
            print(directory + "监测数据-独墅-" + i.split('-')[2])
            df.to_csv(directory + "监测数据-独墅-" + i.split('-')[2], index=False, encoding='utf_8_sig')
        elif i.__contains__('金鸡湖'):
            print(directory + "监测数据-金鸡-" + i.split('-')[2])
            df.to_csv(directory + "监测数据-金鸡-" + i.split('-')[2], index=False, encoding='utf_8_sig')

    for i in outside:
        df = pd.read_csv(directory + i, index_col=None)
        if i.__contains__('独墅湖'):
            print(directory + "监测数据-独墅-" + i.split('-')[2])
            df.to_csv(directory + "监测数据-独墅-" + i.split('-')[2], index=False, encoding='utf_8_sig')
        elif i.__contains__('金鸡湖'):
            print(directory + "监测数据-金鸡-" + i.split('-')[2])
            df.to_csv(directory + "监测数据-金鸡-" + i.split('-')[2], index=False, encoding='utf_8_sig')


def daily_exploring(directory: str):
    # drop_list = ['salinity', 'ammonia nitrogen', 'dissolved oxygen (%Sat)', 'PH value (mv)', 'low frequency water depth (m)']
    for file in os.listdir(directory):
        if file != 'raw_data' and file != 'j-20211021':
            data = pd.read_csv(directory + file, index_col=None)
            data = data.fillna(data.interpolate())
            data['time'] = pd.to_datetime(data['time']).dt.time  # Fix time format
            # cor = correlation(data.drop('time', axis=1))
            # Normal test
            # norm_test(data.drop('time', axis=1))  # Result: Not normal distribution
            # Correlation
            df = data[['phycoprotein', 'chlorophyll', 'pH value', 'temperature', 'dissolved oxygen']]
            df.index = data['time']
            print(file.split('-')[1], ': ')
            print(correlation(df))
            # df.plot()
            # plt.show()


def norm_test(data, res, file):
    cols = ['v', 'chlorophyll', 'conductivity',
            'dissolved oxygen', 'phycoprotein',
            'total dissolved solids', 'turbidity', 'temperature', 'PH value']
    row = [file]
    print(file)
    for col in data.columns.values:
        stat, p = normaltest(data[col])
        print(col + ': ', p)
    # res = res.append(pd.DataFrame(row, columns=cols), ignore_index=True)
    print('\n')
    return  # res


def norm_test_all(directory: str):
    cols = ['v', 'chlorophyll', 'conductivity',
            'dissolved oxygen', 'phycoprotein',
            'total dissolved solids', 'turbidity', 'temperature', 'pH value']
    res = pd.DataFrame(columns=cols)
    for file in os.listdir(directory):
        if file != 'raw_data':
            df = pd.read_csv(directory + file, index_col=None).drop(
                ['time', 'longitude', 'latitude', 'low frequency water depth (m)'], axis=1)
            norm_test(df, res, file)
    res.to_csv(directory + "j-" + "NormTest.csv", index=False, sep=',')


def create_export_name(file_dir, file_name):
    export_name = ''
    date = file_name.split('-')[2]
    if file_name.__contains__('d-inside'):
        export_name = '监测数据-独墅湖内湖-'
    elif file_name.__contains__('d-outside'):
        export_name = '监测数据-独墅湖外湖-'
    elif file_name.__contains__('j-outside'):
        export_name = '监测数据-金鸡湖外湖-'
    elif file_name.__contains__('j-inside'):
        export_name = '监测数据-金鸡湖内湖-'

    return file_dir + export_name + date.split('.')[0] + '.csv'


def change_name(file_dir):
    for file in os.listdir(file_dir):
        df = pd.read_csv(file_dir + file)
        df.rename(
            columns={'longitude': '经度', 'latitude': '纬度', 'time': '时间', 'chlorophyll': '叶绿素', 'conductivity': '电导率',
                     'low frequency water depth (m)': '低频水深(m)', 'dissolved oxygen (%Sat)': '溶解氧(% Sat)',
                     'dissolved oxygen(mg/L)': '溶解氧(mg/L)', 'ammonia nitrogen': '氨氮值', 'salinity': '盐度',
                     'phycoprotein': '藻蛋白', 'total dissolved solids': '总溶解固体', 'turbidity': '浊度',
                     'temperature': '温度', 'PH value': 'PH值', 'PH value (mv)': 'PH值(mv)'}, inplace=True)
        save_name = create_export_name(file_dir, file)
        df.to_csv(save_name, index=False, encoding='utf_8_sig')


# merge_data(file_dir)
# dataset = load_data(file_dir)
# dataset = dataset.fillna(dataset.interpolate()).drop('time', axis=1)
# daily_exploring(file_dir)
# corr = correlation(dataset.drop('time', axis=1))
# norm_test_all(file_dir)
# change_name(file_dir)


attrs = ['longitude', 'latitude', 'time', 'chlorophyll', 'conductivity', 'low frequency water depth (m)',
         'dissolved oxygen (%Sat)',
         'dissolved oxygen(mg/L)', 'ammonia nitrogen', 'salinity', 'phycoprotein', 'total dissolved solids',
         'turbidity', 'temperature',
         'PH value,PH value (mv)']

file_dir = "./datasets/监测数据/"
merge_data(file_dir)
