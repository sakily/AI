import pandas as pd
import h5py
import numpy as np


#1.fetch data
def read_h5_file(code: str,date: str,filepath: str,) -> pd.DataFrame:
    """read h5 file as dataframe"""
    with h5py.File(filepath, mode='r') as h5_reader:
        data = pd.DataFrame(h5_reader[date.replace('-', '')][:])
    data.to_csv(code+'_'+date+'.csv')
    return data

#2.data transform-----------------------------------------------------------------------
def _transform_date_from_h5(date: pd.Series) -> pd.Series:
    function_date = lambda x: '{}-{}-{}'.format(x[:4], x[4:6], x[6:8])
    return date.astype(str).map(function_date)


def _transform_time_from_h5(time: pd.Series) -> pd.Series:
    if (time % 1000 == 0).all():
        function_time = lambda x: '{}:{}:{}'.format(x[:2], x[2:4], x[4:6])
    else:
        function_time = lambda x: '{}:{}:{}.{}'.format(x[:2], x[2:4], x[4:6], x[6:9])
    return time.map(lambda x: function_time(str(x).zfill(9)))


def _transform_snapshot_from_h5(snapshot: pd.DataFrame) -> pd.DataFrame:
    """add, delete and rename columns from h5 snapshot as (old) csv snapshot"""
    new_snapshot = snapshot.copy()

    # add columns: datetime
    # don't add columns: open, prev_close, change_rate, limit_up, limit_down
    new_snapshot['date'] = _transform_date_from_h5(new_snapshot['date'])
    new_snapshot['time'] = _transform_time_from_h5(new_snapshot['time'])
    new_snapshot['datetime'] = pd.to_datetime((new_snapshot['date'] + ' ' + new_snapshot['time']))
    # remove columns: time
    new_snapshot = new_snapshot.drop(columns=['time'], axis=1)

    # rename columns: date -> trading_date
    new_snapshot = new_snapshot.rename(columns={'date': 'trading_date'})

    # unify data form
    price_cols = ['last', 'high', 'low'] + [i + str(j) for i in ['a', 'b'] for j in np.arange(5) + 1]
    new_snapshot[price_cols] = new_snapshot[price_cols] / 10000

    # return new_snapshot.set_index('datetime')
    return new_snapshot



#3.data preprocess-----------------------------------------------------------------------
def del_abr_data(data:pd.DataFrame) -> pd.DataFrame:
    '''delete abnormal data'''
    data[data['a1'] == 0] = np.nan  # 如买一价为0，将数据设为nan
    data[data['b1']==0] = np.nan
    data.dropna(inplace=True)  # 删除数据
    return data

def drop_duplicate(data: pd.DataFrame):
    '''handle duplicates'''
    data.drop_duplicates()
    for i in range(len(data)-1):
        if (data.index[i+1] - data.index[i]).seconds < 2:
            d = data.iloc[i + 1] == data.iloc[i]
            if all(x for x in d):
                data.drop(data.index[i])
    return data


def resample_data(freq: str,data: pd.DataFrame, date_list: list, code_list: list) -> pd.DataFrame:
    '''resample the data to certain frequency'''
    l = []
    for code in code_list:
        d1 = data[data.order_book_id == code].resample(freq).bfill()
        for date in date_list:
            start_1 = pd.to_datetime(date + '091500')
            end_1 = pd.to_datetime(date + '113000')
            start_2 = pd.to_datetime(date + '130000')
            end_2 = pd.to_datetime(date + '150000')
            l.append(d1.loc[start_1:end_1])
            l.append(d1.loc[start_2:end_2])
    data_bar = pd.concat(l)
    return data_bar



def get_trading_period(data: pd.DataFrame, date_list: list, code_list: list) -> pd.DataFrame:
    '''get data within trading period(9：30-11：30，13：00-15：00)'''
    l = []
    for code in code_list:
        d1 = data[data.order_book_id == code]
        for date in date_list:
            start = pd.to_datetime(date + '093000')
            end = pd.to_datetime(date + '150000')
            l.append(d1.loc[start:end])
    data = pd.concat(l)
    return data

def get_min(datetime):
    return(datetime.strftime('%Y-%m-%d %H:%M:%S'))

def reset_time_format(data: pd.DataFrame) -> pd.DataFrame:
    data = data.reset_index(level='datetime',drop=False)
    data['datetime'] = data['datetime'].apply(get_min)
    return data.set_index('datetime')