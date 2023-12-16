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
            plt.figure(figsize=(12, 6))
            plt.plot(data.index[-len(predictions):], data['Close'][-len(predictions):], label='Real Close Price',
                     color='blue')
            plt.plot(data.index[-len(predictions):], predictions, label='Predicted Close Price', color='red',
                     linestyle='dashed')
            plt.title(title)
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()
            plt.show()
        else:
            print("Invalid data. Unable to plot.")

    @staticmethod
    def plot_decomposition(decomposition, title: str = ""):
        trend_estimate = decomposition.trend
        seasonal_estimate = decomposition.seasonal
        residual_estimate = decomposition.resid
        observed = decomposition.observed

        fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
        plt.title(title)
        fig.set_figheight(10)
        fig.set_figwidth(20)
        # First plot to the Original time series
        axes[0].plot(observed, label='Original')
        axes[0].legend(loc='upper left')
        # second plot to be for trend
        axes[1].plot(trend_estimate, label='Trend')
        axes[1].legend(loc='upper left')
        # third plot to be Seasonality component
        axes[2].plot(seasonal_estimate, label='Seasonality')
        axes[2].legend(loc='upper left')
        # last last plot to be Residual component
        axes[3].plot(residual_estimate, label='Residuals')
        axes[3].legend(loc='upper left')
        plt.legend()
