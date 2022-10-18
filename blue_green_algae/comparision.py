import os

import pandas as pd


def create_export_name(file_dir, file_name):
    """
    modify principles according to file name
    :param file_dir:
    :param file_name: 监测数据-xx湖x湖-date.csv
    :return:
    """
    export_name = ''
    date = file_name.split('_')[2]
    if file_name.__contains__('dushu_inner'):
        export_name = '监测数据-独墅湖内湖-'
    elif file_name.__contains__('dushu_outer'):
        export_name = '监测数据-独墅湖外湖-'
    elif file_name.__contains__('jinji_outer'):
        export_name = '监测数据-金鸡湖外湖-'
    elif file_name.__contains__('jinji_inner'):
        export_name = '监测数据-金鸡湖内湖-'

    return file_dir + export_name + date.split('.')[0] + '.csv'


def merge_access_data(file_dir, file_name):
    """
    SQL for generate the xlsx file from accessDB:
        SELECT parameterdata.*
        FROM CruiseTask INNER JOIN ParameterData ON CruiseTask.id = ParameterData.taskId
        WHERE cruisetask.id = {};
    :param file_name:
    :return: exported csv similar to 启澄
    """
    # read .xlsx file
    df = pd.read_excel(file_dir + file_name)
    drop_list = ['id', 'taskId', 'parameterId']

    # drop useless attributes, sort by time and reset index
    df = df.drop(drop_list, axis=1) \
        .sort_values(by='reportTime') \
        .reset_index(drop=True)

    # rename attributes and find all parameter names
    df.rename(columns={'lon': '经度', 'lat': '纬度', 'reportTime': '时间'}, inplace=True)
    attrs = list(df['parameterName'].unique())

    # inner join by report time
    new_df = df[['时间', '经度', '纬度']].drop_duplicates().dropna().reset_index(drop=True)
    for i, attr in enumerate(attrs):
        a = df.where(df['parameterName'] == attr).dropna()
        a.rename(columns={'value': attr}, inplace=True)
        a = a[['时间', attr]].drop_duplicates(subset='时间').dropna().reset_index(drop=True)

        new_df = pd.merge(new_df, a, how='inner', on='时间')
        new_df.rename(columns={'氨氮': '氨氮值', '深度(m)': '低频水深(m)'}, inplace=True)

        # save as csv
        save_dir = './datasets/temp1/'
        export_name = create_export_name(save_dir, file_name)
        new_df.to_csv(export_name, index=False, encoding='utf_8_sig')


def merge_access_data_all(file_dir):
    """
    :param file_dir: './datasets/temp/'
    :return:
    """
    for file in os.listdir(file_dir):
        # print(create_export_name('./datasets/temp1/', file))
        merge_access_data(file_dir, file)


def delete_rows(file_dir):
    for file in os.listdir(file_dir):
        df = pd.read_csv(file_dir + file, header=2)
        df['时间'] = pd.to_datetime(df['时间'])
        df.to_csv(file_dir + file, index=False, encoding='utf_8_sig')
        # print(file)


def get_mean_stat(file_dir):
    attrs = ['溶解氧(mg/L)', '总溶解固体', '温度', '藻蛋白',
             '盐度', '浊度', '电导率', '叶绿素', 'PH值']
    stat = pd.DataFrame()
    for file in os.listdir(file_dir):
        loc_date = file.split('-')[1] + file.split('-')[2].split('.')[0]
        try:
            df = pd.read_csv(file_dir + file).dropna()
            desc = df[attrs].describe()
            mean_ = desc.iloc[1, :]
            mean_.name = loc_date
            stat = stat.append(mean_)
            # print(file + ' finished')
        except (KeyError):
            print(file + ' failed')
    stat.to_csv('./datasets/mean_stat1.csv', index=True, encoding='utf_8_sig')


def split_data():
    file_name = './datasets/监测数据/监测数据-独墅_金鸡-20220701.csv'
    dushu = '2022-07-01 11:34:08'
    jinji = '2022-07-01 11:43:50'
    date = file_name.split('-')[2]
    d_name = './datasets/监测数据/监测数据-' + file_name.split('-')[1].split('_')[0] + '-' + date
    j_name = './datasets/监测数据/监测数据-' + file_name.split('-')[1].split('_')[1] + '-' + date
    df = pd.read_csv(file_name)

    d_index = df[df['时间'] == dushu].index.values[0]
    j_index = df[df['时间'] == jinji].index.values[0]
    df_dushu = df.iloc[0: d_index].reset_index(drop=True).dropna()
    df_jinji = df.iloc[j_index:].reset_index(drop=True).dropna()

    df_dushu.to_csv(d_name, index=False, encoding='utf_8_sig')
    df_jinji.to_csv(j_name, index=False, encoding='utf_8_sig')


file_dir = './datasets/监测数据1/'
get_mean_stat(file_dir)
