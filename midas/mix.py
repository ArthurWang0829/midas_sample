import datetime
import re
import pandas as pd
import numpy as np


def mix_freq(lf_data, hf_data, xlag, ylag=0, horizon=0, start_date=None, end_date=None):
    """
    Set up data for mixed-frequency regression

    Args:
        lf_data (Series): Low-frequency time series
        hf_data (Series): High-frequency time series
        xlag (int or str): Number of high frequency lags
        ylag (int or str): Number of low-frequency lags
        horizon (int):
        start_date (date): Date on which to start estimation
        end_date (date); Date on which to end estimation

    Returns:

    """

    ylag = calculate_lags(ylag, lf_data)
    xlag = calculate_lags(xlag, hf_data)

    min_date_y = lf_data.index[ylag]
    min_date_x = hf_data.index[xlag + horizon]

    if min_date_y < min_date_x:
        min_date_y = next(d for d in list(lf_data.index) if d > min_date_x)

    if (start_date is None) or (start_date < min_date_y):
        start_date = min_date_y
    if end_date is None:
        end_date = lf_data.index[-2]

    max_date = lf_data.index[-1]
    if max_date > hf_data.index[-1]:
        max_date = next(d for d in reversed(list(lf_data.index)) if d < hf_data.index[-1])

    if end_date > max_date:
        end_date = max_date

    forecast_start_date = lf_data.index[lf_data.index.get_loc(end_date) + 1]

    ylags = None
    if ylag > 0:
        # N.B. ylags will be a dataframe because there can be more than 1 lag
        ylags = pd.concat([lf_data.shift(l) for l in range(1, ylag + 1)], axis=1)

    x_rows = []
    x_s = []
    for lfdate in lf_data.loc[start_date:max_date].index:
        start_hf = hf_data.index.get_loc(lfdate, method='bfill')  # @todo Find a more efficient way
        x_rows.append(hf_data.iloc[start_hf - horizon: start_hf - xlag - horizon: -1].values.reshape(-1))

    x = pd.DataFrame(data=x_rows, index=lf_data.loc[start_date:max_date].index)
    x.columns = ['lag' + str(i) for i in x.columns.tolist()]
    x1 = pd.DataFrame(data=x_s, index=lf_data.loc[start_date:max_date].index)
    return pd.concat([
        lf_data.loc[start_date:end_date],
        # ylags.loc[start_date:end_date] if ylag > 0 else None,
        x.loc[start_date:end_date],
        # lf_data[forecast_start_date:max_date],
        # ylags[forecast_start_date:max_date] if ylag > 0 else None,
        # x.loc[forecast_start_date:max_date]
    ], axis=1)


def calculate_lags(lag, time_series):
    if isinstance(lag, str):
        return parse_lag_string(lag, data_freq(time_series)[0])
    else:
        return lag


def data_freq(time_series):
    """
    Determine frequency of given time series

    Args:
        time_series (Series): Series with datetime index

    Returns:
        string: frequency specifier
    """
    try:
        freq = time_series.index.freq
        return freq.freqstr or pd.infer_freq(time_series.index)
    except AttributeError:
        return pd.infer_freq(time_series.index)


def parse_lag_string(lag_string, freq):
    """
    Determine number of lags from lag string

    Args:
        lag_string: String indicating number of lags (eg, "3M", "2Q")
        freq (string): Frequency of series to be lagged

    Returns:

    """

    freq_map = {
        'd': {'m': 30, 'd': 1},
        'b': {'m': 22, 'b': 1},
        'm': {'q': 3, 'm': 1},
        'q': {'y': 4},
        'a': {'y': 1}
    }

    m = re.match('(\d+)(\w)', lag_string)

    duration = int(m.group(1))
    period = m.group(2).lower()

    return duration * freq_map[freq.lower()][period]


def get_corr_ths(corr, ths=0.5):
    return corr[corr > ths]

def max_coef_lags(corr):
    corr = corr.iloc[0]
    max_coef = corr.replace(1, 0).max()
    max_lags = corr.index[corr.replace(1, 0).argmax()]
    return [max_coef, max_lags]

# if __name__ == '__main__':

import os

print(os.getcwd())
os.chdir('../')
lf_data = pd.read_csv('tests/data/gdp.csv', parse_dates=['DATE'])
lf_data.set_index('DATE', inplace=True)

hf_data = pd.read_csv('tests/data/pay.csv', parse_dates=['DATE'])
hf_data.set_index('DATE', inplace=True)

lf_g = np.log(1 + lf_data.pct_change()).dropna() * 100.
hf_g = np.log(1 + hf_data.pct_change()).dropna() * 100.

# df = pd.concat([lf_g, hf_g], join='inner')
df = mix_freq(lf_g, hf_g, 9, 1, 0, datetime.date(1996,1,1), datetime.date(2008, 7, 1))
# df = mix_freq(lf_data, hf_data, 9, 1, 0, datetime.date(1996, 1, 1), datetime.date(2008, 1, 1))
print(df)
print(max_coef_lags(df.corr()))
import matplotlib.pyplot as plt
import seaborn as sns

# plt.subplots(figsize=(9, 9))
# sns.heatmap(df.corr(), annot=True, vmax=1, square=True, cmap='Blues')
# plt.show()
