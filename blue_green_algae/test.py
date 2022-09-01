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
    file1 = pd.read_csv(directory + 'raw_data/' + f1, index_col=None)
    file2 = pd.read_csv(directory + 'raw_data/' + f2, index_col=None)
    file = [file1, file2]
    drop_list = ['salinity', 'ammonia nitrogen', 'dissolved oxygen (%Sat)', 'PH value (mv)']
    train = pd.concat(file).drop(drop_list, axis=1)
    train.to_csv(directory + "d-" + f1.split('-')[2], index=False, sep=',')


def merge_data(directory: str):
    # raw data: ./datasets/jinji_lake/raw_data
    # merged data: ./dataset/jinji_lake
    inside = []
    outside = []
    for file in os.listdir(directory + 'raw_data/'):
        if file.__contains__('inside'):
            inside.append(file)
        elif file.__contains__('outside'):
            outside.append(file)
    for i in inside:
        for o in outside:
            if o.split('-')[2] == i.split('-')[2]:
                concat_csv(i, o, directory)
                break


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


file_dir = "./datasets/dushu_jinji_lake/"
# merge_data(file_dir)
# dataset = load_data(file_dir)
# dataset = dataset.fillna(dataset.interpolate()).drop('time', axis=1)
daily_exploring(file_dir)
# corr = correlation(dataset.drop('time', axis=1))
# norm_test_all(file_dir)
