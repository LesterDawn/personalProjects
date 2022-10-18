# -*- coding: utf-8 -*-
import scipy.stats as stats
from scipy.stats import spearmanr, kstest
import pandas as pd
from sklearn import preprocessing
import scipy.signal
import matplotlib.pyplot as plt
import os
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

pd.set_option('display.max_columns', 1000)


def set_ylabel(attr):
    """
    Add units to the plots
    """
    if attr == 'chlorophyll':
        plt.ylabel('chlorophyll (\u03bcg/L)', fontsize=18)
    elif attr == 'temperature':
        plt.ylabel(u'temperature (\u00B0C)', fontsize=18)
    elif attr == 'dissolved oxygen':
        plt.ylabel('dissolved oxygen (mg/L)', fontsize=18)
    else:
        plt.ylabel(attr, fontsize=18)


def drop_outliers(df: pd.DataFrame):
    """
    Delete outliers using z-score normalization
    """
    # df = df.drop('time', axis=1)
    attributes = ['chlorophyll', 'temperature',
                  'pH value', 'dissolved oxygen', 'phycoprotein']
    df = df[attributes]

    # Drop invalid values
    df = df[df['chlorophyll'] > 0]
    df = df[df['dissolved oxygen'] > 0]

    # Get z-score norm
    df_norm = (df - df.mean()) / (df.std())

    # Only maintain the data that is less than 3
    tmp = df[abs(df_norm[:]) <= 3].dropna()
    return tmp


def plot_line(df, df_norm, attr: str, filename: str, plt_type: str):
    # smooth the line of attr using savgol filter
    df_norm['phycoprotein'] = pd.DataFrame(scipy.signal.savgol_filter(df_norm['phycoprotein'], 53, 3))
    df_norm[attr] = pd.DataFrame(scipy.signal.savgol_filter(df_norm[attr], 53, 3))

    # # ks test, all data are not normal distribution
    # print(kstest(df['phycoprotein'], cdf="norm"))
    # print(kstest(df[attr], cdf="norm"))
    if plt_type == 'lineChart':
        plt.figure(figsize=(20, 8), dpi=80)
        plt.plot(df_norm.index, df_norm['phycoprotein'], label='phycocyanin', color='red')
        plt.plot(df_norm.index, df_norm[attr], label=attr, color='green')
        plt.xlabel('Sample index', fontsize=18)
        plt.ylabel('Z-score', fontsize=18)
        plt.legend(loc='upper left', fontsize=16)
        plt.tick_params(labelsize=18)
    elif plt_type == 'scatter':
        plt.figure()
        df = drop_outliers(df)
        plt.scatter(df['phycoprotein'], df[attr], s=1)
        plt.xlabel('phycocyanin (million/L)', fontsize=18)
        set_ylabel(attr)
        plt.tick_params(labelsize=16)
    title = filename.split('.')[0] + '_phycocyanin-' + attr
    # + '\nPearson correlation coefficient: %.4f' % df['phycoprotein'].corr(df[attr], method='pearson'
    plt.title(u'Spearman\'s \u03c1: %.4f***'
              % spearmanr(df_norm['phycoprotein'], df_norm[attr])[0], fontsize=18)
    # print(title + ' p-value: %f\nAmount of samples: %d' % (spearmanr(df_norm['phycoprotein'], df_norm[attr])[1],
    # len(df_norm['phycoprotein'])))
    if plt_type == 'scatter':
        title += '_scatter'
    # Save figures
    print('Amount of samples: %d' % len(df['phycoprotein']))
    fn = './corr_figures/' + title + '.png'
    plt.tight_layout()
    # plt.savefig(fn)
    plt.close()

    # plt.show()


def get_corr_plot(directory, filename: str, plt_type: str, norm: str):
    """
    :param filename:
    :param directory:
    :param norm: 'min-max' or 'z-score' or 'none'
    :param plt_type: 'scatter' or 'lineChart'
    """
    # Load data
    df = pd.read_csv(directory + filename, index_col=None)
    attributes = ['time', 'chlorophyll', 'temperature',
                  'pH value', 'dissolved oxygen', 'phycoprotein']
    df = df[attributes]
    df['time'] = pd.to_datetime(df['time']).dt.time
    # Unit transform for phycocprotein: 10000/L -> million/L
    df['phycoprotein'] = df['phycoprotein'] / 100

    # Normalization, min-max / Z-score
    attributes.remove('time')
    df_norm = pd.DataFrame()
    if norm == 'min-max':  # min-max is useless*********
        df[attributes] = min_max_norm(df[attributes])
        filename = 'm_' + filename
    elif norm == 'z-score':
        df_norm = z_score_norm(df[attributes])
        filename = filename
    else:
        print('No normalization\n')

    # Get plot
    # attributes.remove('phycoprotein')
    for attr in attributes:
        if attr != 'phycoprotein':
            plot_line(df, df_norm, attr, filename, plt_type)
    return df, df_norm


def min_max_norm(df):
    df = drop_outliers(df)  # drop outliers using z-score first
    # reset index since outliers and invalid values are dropped
    df = df.reset_index(drop=True)
    df_norm = (df - df.min()) / (df.max() - df.min())
    return df_norm


def z_score_norm(df):
    df = drop_outliers(df)  # drop outliers using z-score first
    # reset index since outliers and invalid values are dropped
    df = df.reset_index(drop=True)
    df_norm = (df - df.mean()) / (df.std())
    tmp = df_norm[abs(df_norm[:]) <= 3].dropna()
    # reset index since outliers are dropped
    tmp = tmp.reset_index(drop=True)
    return tmp


file_dir = "./datasets/dushu_lake/"
# file_name = "d-20210929.csv"
# data = get_corr_plot(file_dir, file_name, norm='z-score')
#
# Get lines of all dates
data = pd.DataFrame()
for file in os.listdir(file_dir):  # and file != 'j-all.csv' and file != 'd-all.csv' \
    if file != 'raw_data' and file != 'j-20211021.csv' \
            and file != 'processed_d-all.csv' and file != 'processed_j-all.csv' \
            and file != 'j-stat.csv' and file != 'd-stat.csv' and file != 'd_j-stat.csv':
        data = get_corr_plot(file_dir, file, norm='z-score', plt_type='lineChart')

# data1 = pd.read_csv(file_dir + 'd-20210901.csv')
