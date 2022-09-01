import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
import scipy.signal
import matplotlib.pyplot as plt
import os


def drop_outliers(df: pd.DataFrame):
    """
    Delete outliers using z-score normalization
    """
    df = df.drop('time', axis=1)
    attributes = ['chlorophyll', 'temperature', 'pH value', 'dissolved oxygen', 'phycoprotein']
    df = df[attributes]

    # Drop invalid values
    df = df[df['chlorophyll'] > 0]
    df = df[df['dissolved oxygen'] > 0]

    # Get z-score norm
    df_norm = (df - df.mean()) / (df.std())

    # Only maintain the data whose abs is less than 3
    tmp = df[abs(df_norm[:]) <= 3].dropna()
    return tmp


def get_stat(df: pd.DataFrame, stat: pd.DataFrame):
    """
    get mean, std, min, max from one date data
    """
    desc = df.describe()
    stat = stat.append(desc.iloc[1:4, :])
    stat = stat.append(desc.iloc[7, :])
    # print(desc.iloc[1, :])
    return stat


def get_stat_all(directory: str):
    """
    get stat of all dates
    """
    stat = pd.DataFrame()
    for file in os.listdir(file_dir):
        if file != 'raw_data' and file != 'j-20211021.csv' \
                and file != 'j-all.csv' and file != 'd1-all.csv' \
                and file != 'j-stat.csv' and file != 'd-stat.csv':  #
            df = pd.read_csv(directory + file, index_col=None)
            df_processed = drop_outliers(df)
            if file == 'd-all.csv':
                df_processed.to_csv(file_dir + 'processed_' + file, index=False)
            # print(file)
            stat = get_stat(df_processed, stat)
    return stat


file_dir = "./datasets/dushu_lake/"
stat = get_stat_all(file_dir)
# stat.to_csv(file_dir+'stat.csv')
