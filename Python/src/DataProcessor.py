import pandas as pd

class DataProcessor:
    @staticmethod
    def process_data(data_df: pd.DataFrame) -> pd.DataFrame:
        # Filling NaN values with the previous day's closing price
        data_df['Adj Close'].fillna(method='ffill', inplace=True)

        # Dropping rows with any remaining NaN values
        data_df.dropna(inplace=True)

        return data_df