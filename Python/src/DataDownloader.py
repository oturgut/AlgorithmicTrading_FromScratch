from typing import List, Optional
import pandas as pd
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import os
from DataProcessor import DataProcessor

class DataDownloader:
    @staticmethod
    def download_data(
        symbols: List[str],
        start_date: str,
        end_date: str,
        save_path: Optional[str] = "data",
    ) -> dict:
        downloaded_data = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    DataDownloader._download_symbol_data,
                    symbol,
                    start_date,
                    end_date,
                    save_path,
                ): symbol
                for symbol in symbols
            }

            for future, symbol in futures.items():
                try:
                    data_df = future.result()
                    downloaded_data[symbol] = data_df
                except Exception as e:
                    print(f"Error downloading data for {symbol}: {str(e)}")

        return downloaded_data

    @staticmethod
    def _download_symbol_data(
        symbol: str, start_date: str, end_date: str, save_path: Optional[str]
    ) -> pd.DataFrame:
        data_df = None

        # Check if the file already exists
        if save_path:
            symbol_filename = f"{save_path}/{symbol}_stock_data_{start_date}_{end_date}.pkl"
            if os.path.exists(symbol_filename):
                try:
                    data_df = pd.read_pickle(symbol_filename)
                    print(f"Data for {symbol} loaded from {symbol_filename}")
                    return data_df
                except Exception as e:
                    print(f"Error loading data for {symbol} from {symbol_filename}: {str(e)}")

        try:
            # Download data if the file does not exist or could not be loaded
            data_df = yf.download(symbol, start=start_date, end=end_date)

            # Process the data before saving
            data_df = DataProcessor.process_data(data_df)

            if save_path:
                data_df.to_pickle(symbol_filename)
                print(f"Data for {symbol} saved to {symbol_filename}")

        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")

        return data_df
