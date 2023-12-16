from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd


class DataTrend:
    def __init__(self, data, column_name, period=30):
        self.data = data
        self.column_name = column_name
        self.period = period

    def decompose(self, model='multiplicative'):
        decomposition = seasonal_decompose(self.data[self.column_name], model=model, period=self.period)
        return decomposition

