"""
=====================================================================================
Модель Holt-Winters Exponential Smoothing.
=====================================================================================

****** Описание модуля ******
Модуль имеет две функции serialize и performance.
Функция serialize вызывается для создания pickle file и прогноза данных.
Функция performance служит визуализатором графиков actual data и forecasted.

"""
import matplotlib
matplotlib.use('TKAgg')
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
from apiclass import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error as mse
from pylab import rcParams
import sys
import warnings
import pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def serialize(param_data, filename, train_idx=slice(97000,100000), test_idx=slice(100000,103000), seasonal_periods=314):
    """Creates pickle file of the model


        Parameters
        ----------
        param_data : a one-dimensional Pandas dataframe without time indexes

        filename : The name of the pickle file to be created
            A str format string: ex. 'Model_saved'

        train_idx : Training data indices in slice format.

        test_idx : Data indices for the forecast
            "slice" formatted indices.
        seasonal_periods : The period of the data.
            An integer number.
        Returns
        -------
        """
    train, test = param_data.iloc[train_idx], param_data.iloc[test_idx]
    param_data.index.freq = 'sec'
    model = ExponentialSmoothing(train,seasonal='add',seasonal_periods=seasonal_periods).fit()
    pred = model.predict(start=test.index[0], end=test.index[-1])

    with open('%s.pickle' % filename, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(mse(pred,param_data.iloc[test.index[0]-1:test.index[-1]]))


def performance(param_data, filename, test_idx):
    """Loads an existing pickle file, makes prediction via .predict method and then plots the input and forecasted data

        Parameters
        ----------
        param_data: a one-dimensional Pandas dataframe without time indexes

        filename: The name of the pickle file of the model(only name, without ".pickle")

        test_idx: Data indices for the forecast
            "slice" formatted indices.
    """
    with open('%s.pickle' % filename, 'rb') as handle:
        model1 = pickle.load(handle)
        pred = model1.predict(start=param_data.iloc[test_idx].index[0], end=param_data.iloc[test_idx].index[-1])#param_data.index[-1])

    plt.rcParams["figure.figsize"] = 20,5
    plt.plot(param_data.iloc[test_idx], label='Test')
    plt.plot(pred, label='Holt-Winters')
    plt.legend(loc='best')
    plt.show()
    print(mse(pred,param_data.iloc[param_data.iloc[test_idx].index[0]-1:param_data.iloc[test_idx].index[-1]]))


if __name__ == "__main__":
    df = pd.read_csv("db1.autogen.Thread-Gen-1.2018-11-07-10-12.csv")
    df = df["Thread-Gen-1.mean_sinusoid"]
    rcParams['figure.figsize'] = 20, 5
    plt.plot(df)
    print(df.shape)

    i = 115
    train_idx=slice(0,i)
    test_idx=slice(i,350)
    serialize(df, 'model_saved', train_idx, test_idx, seasonal_periods=39)
    performance(df, 'model_saved', test_idx)