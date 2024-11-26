import warnings
import pandas as pd
import numpy as np


from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')


def reduce_mem_usage(df):
    """ Функция для оптимизации использования памяти DataFrame (inplace). """

    # Расчет начального использования памяти -
    start_memory = df.memory_usage().sum() / 1024 ** 2

    # Создание словарей с диапазонами для каждого типа чисел
    int_type_dict = {
        (np.iinfo(np.int8).min, np.iinfo(np.int8).max): np.int8,
        (np.iinfo(np.int16).min, np.iinfo(np.int16).max): np.int16,
        (np.iinfo(np.int32).min, np.iinfo(np.int32).max): np.int32,
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max): np.int64,
    }

    float_type_dict = {
        (np.finfo(np.float16).min, np.finfo(np.float16).max): np.float16,
        (np.finfo(np.float32).min, np.finfo(np.float32).max): np.float32,
        (np.finfo(np.float64).min, np.finfo(np.float64).max): np.float64,
    }

    # Обрабатываем каждый столбец в DataFrame
    for column in df.columns:
        col_type = df[column].dtype

        if np.issubdtype(col_type, np.integer):
            c_min = df[column].min()
            c_max = df[column].max()
            dtype = next((v for k, v in int_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                df[column] = df[column].astype(dtype)
        elif np.issubdtype(col_type, np.floating):
            c_min = df[column].min()
            c_max = df[column].max()
            dtype = next((v for k, v in float_type_dict.items() if k[0] <= c_min and k[1] >= c_max), None)
            if dtype:
                df[column] = df[column].astype(dtype)

    # Расчет конечного использования памяти
    end_memory = df.memory_usage().sum() / 1024 ** 2
    return df

def filter_data(df):
    #удаленине лишних признаков
    coloumns_for_drop = ['enc_paym_0',
                         'enc_paym_1',
                         'enc_paym_2',
                         'enc_paym_3',
                         'enc_paym_4',
                         'enc_paym_5',
                         'enc_paym_6',
                         'enc_paym_7',
                         'enc_paym_8',
                         'enc_paym_9',
                         'enc_paym_10',
                         'enc_paym_11',
                         'enc_paym_12',
                         'enc_paym_13',
                         'enc_paym_14',
                         'enc_paym_15',
                         'enc_paym_16',
                         'enc_paym_17',
                         'enc_paym_18',
                         'enc_paym_19',
                         'enc_paym_20',
                         'enc_paym_21',
                         'enc_paym_22',
                         'enc_paym_23',
                         'enc_paym_24']
    df = df.copy()
    df = df.drop(coloumns_for_drop, axis=1)
    return df


def merge_flag(df):
    #группировка по id
    target = pd.read_csv("data/train_target.csv")
    df = pd.merge(left=df.groupby('id').agg('mean'), right=target, on='id', how='left')
    df = df.drop(['flag', 'id'], axis=1)
    df = df.reset_index(drop=True)
    print('merged')
    return df

def imput_median(df):
    #обработка пропущенных значений
    df.fillna(df.median(), inplace=True)
    df = df.reset_index(drop=True)
    print('filled')
    return df


def encoder(df):
    #OHE кодирование
    print('start enc')
    data_ohe_list = ['pre_till_fclose',
                     'pre_till_pclose',
                     'pre_fterm',
                     'pre_pterm',
                     'pre_since_confirmed',
                     'pre_since_opened',
                     'is_zero_loans5',
                     'is_zero_loans530',
                     'is_zero_loans3060',
                     'is_zero_loans6090',
                     'is_zero_loans90',
                     'is_zero_util',
                     'is_zero_over2limit',
                     'is_zero_maxover2limit',
                     'pclose_flag',
                     'fclose_flag',
                     'pre_loans5',
                     'pre_loans530',
                     'pre_loans3060',
                     'pre_loans6090',
                     'pre_loans90',
                     'enc_loans_account_holder_type',
                     'enc_loans_credit_status',
                     'enc_loans_credit_type',
                     'enc_loans_account_cur',
                     'pre_loans_credit_limit',
                     'pre_loans_next_pay_summ',
                     'pre_loans_outstanding',
                     'pre_loans_total_overdue',
                     'pre_loans_max_overdue_sum',
                     'pre_loans_credit_cost_rate',
                     'pre_util',
                     'pre_over2limit',
                     'pre_maxover2limit',
                     'flag_paym0',
                     'flag_paym1',
                     'flag_paym2',
                     'flag_paym3',
                     'flag_paym4',
                     'flag_paym5',
                     'flag_paym6',
                     'flag_paym7',
                     'flag_paym8',
                     'flag_paym9',
                     'flag_paym10',
                     'flag_paym11',
                     'flag_paym12',
                     'flag_paym13',
                     'flag_paym14',
                     'flag_paym15',
                     'flag_paym16',
                     'flag_paym17',
                     'flag_paym18',
                     'flag_paym19',
                     'flag_paym20',
                     'flag_paym21',
                     'flag_paym22',
                     'flag_paym23',
                     'flag_paym24',
                     'is_zero_loans',
                     'all_is_closed']
    data_ohe = df[data_ohe_list]
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', dtype='int8')
    ft = ohe.fit_transform(data_ohe)
    df_ft = pd.DataFrame(ft, columns=ohe.get_feature_names_out())
    df = pd.concat([df, df_ft], axis=1)
    df = df.drop(data_ohe_list, axis=1)
    print('stop enc')
    return df


def new_features(df):
    #создание новых признаков
    df['all_is_closed'] = df['pre_loans_outstanding'].apply(lambda x: 1 if x == 0 else 0)
    df['is_zero_loans'] = df['is_zero_loans5'].apply(lambda x: 1 if x == 1 else 0) & df['is_zero_loans530'].apply(
        lambda x: 1 if x == 1 else 0) & df['is_zero_loans3060'].apply(lambda x: 1 if x == 1 else 0) & df[
                              'is_zero_loans6090'].apply(lambda x: 1 if x == 1 else 0) & df['is_zero_loans90'].apply(
        lambda x: 1 if x == 1 else 0)
    df['flag_paym0'] = df['enc_paym_0'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym1'] = df['enc_paym_1'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym2'] = df['enc_paym_2'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym3'] = df['enc_paym_3'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym4'] = df['enc_paym_4'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym5'] = df['enc_paym_5'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym6'] = df['enc_paym_6'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym7'] = df['enc_paym_7'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym8'] = df['enc_paym_8'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym9'] = df['enc_paym_9'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym10'] = df['enc_paym_10'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym11'] = df['enc_paym_11'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym12'] = df['enc_paym_12'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym13'] = df['enc_paym_13'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym14'] = df['enc_paym_14'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym15'] = df['enc_paym_15'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym16'] = df['enc_paym_16'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym17'] = df['enc_paym_17'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym18'] = df['enc_paym_18'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym19'] = df['enc_paym_19'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym20'] = df['enc_paym_20'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym21'] = df['enc_paym_21'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym22'] = df['enc_paym_22'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym23'] = df['enc_paym_23'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    df['flag_paym24'] = df['enc_paym_24'].apply(lambda x: 1 if x == 0 or x == 1 else 0)
    return df


def boundaries_dataframe(df):
    #обработка выбросов
    bounds = dict()

    def calculate_iqr_boundaries(series):
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

    list_df = list(df)

    for i in range(len(list_df)):
        if (df[list_df[i]].dtypes == 'int64' or df[list_df[i]].dtypes == 'float64' or df[
            list_df[i]].dtypes == 'float32' or df[list_df[i]].dtypes == 'int32') and df[list_df[i]].nunique() > 10:
            bound = calculate_iqr_boundaries(df[list_df[i]])
            df_1 = df[(df[list_df[i]] >= bound[0]) & (df[list_df[i]] <= bound[1])]
            df_1_outlier = (df[list_df[i]] < bound[0]) | (df[list_df[i]] > bound[1])
            df_1_outlier_min = (df[list_df[i]] < bound[0])
            df_1_outlier_max = (df[list_df[i]] > bound[1])

            # print(df_1_outlier)
            if df[list_df[i]].min() < bound[0]:
                df.loc[df_1_outlier_min, str(list_df[i])] = bound[0]
            if df[list_df[i]].max() > bound[1]:
                df.loc[df_1_outlier_max, str(list_df[i])] = bound[1]
            bounds[list_df[i]] = bound

    df = df.reset_index(drop=True)
    return df        