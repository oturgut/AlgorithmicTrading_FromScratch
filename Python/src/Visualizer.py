import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_data(data: pd.DataFrame, title: str):
        if data is not None and not data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(data['Adj Close'], label='Adjusted Close Price', color='blue')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        else:
            print("Invalid data. Unable to plot.")

    @staticmethod
    def plot_predictions(data: pd.DataFrame, predictions: pd.Series, title: str = ""):
        if data is not None and not data.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(data['Close'], label='Real Close Price', color='blue')
            plt.plot(data.index, predictions, label='Predicted Close Price', color='red', linestyle='dashed')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        else:
            print("Invalid data. Unable to plot.")