from datetime import datetime
from DataDownloader import DataDownloader
from Visualizer import Visualizer
from LinearRegressionModel import LinearRegressionModel
from ModelTrainer import ModelTrainer
from DataTrend import DataTrend
import pandas as pd


def main():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2023-11-01"
    save_path = '/Users/ozhanturgut/Documents/GitHub/AlgorithmicTrading_FromScratch/Python/data'
    model_path = '/Users/ozhanturgut/Documents/GitHub/AlgorithmicTrading_FromScratch/Python/model/lr_test.pkl'

    data_downloader = DataDownloader()
    downloaded_data = data_downloader.download_data(
        symbols, start_date, end_date, save_path=save_path
    )

    # Choose the model (you can replace this with your custom model)
    model = LinearRegressionModel()
    visualizer = Visualizer()

    # Assume data_df and target_column are available from your data
    # You can replace this with your actual data loading and preprocessing
    data_df = pd.concat([downloaded_data[symbol].reset_index(level='Date', drop=True) for symbol in symbols[0:1]],
                        keys=symbols, names=['Symbol'])
    # Debugging purposes
    #print(data_df.head())
    target_column = 'Adj Close'

    # Data trends
    for symbol in symbols:
        trend_analyzer = DataTrend(data=downloaded_data[symbol], column_name=target_column, period=30)

        # Perform multiplicative decomposition
        multiplicative_decomposition = trend_analyzer.decompose(model='multiplicative')
        visualizer.plot_decomposition(multiplicative_decomposition, symbol + ' - Multiplicative')

        # Perform additive decomposition
        additive_decomposition = trend_analyzer.decompose(model='additive')
        visualizer.plot_decomposition(additive_decomposition, symbol + ' - Additive')


    # Train the model
    rolling_window_size = 20  # Adjust the window size as needed
    trainer = ModelTrainer(model, data_df=data_df, target_column=target_column, retrain=True,
                           rolling_window_size=rolling_window_size)

    # Save the model
    trainer.save_model(model_path)

    # Or load a pre-trained model
    # trainer = ModelTrainer(model, retrain=False)
    # trainer.load_model('your_pretrained_model_path')

    # Use the trained model to make predictions on another stock (e.g., MSFT)
    new_data_df = downloaded_data["GOOGL"]
    # Drop the 'Date' column from new_data_df
    new_data_df = new_data_df.reset_index(level='Date', drop=True)
    #print(new_data_df.head())

    # Make predictions using a rolling window
    predictions = []
    for i in range(len(new_data_df) - rolling_window_size):
        window_data = new_data_df.iloc[i:i + rolling_window_size]
        window_predictions = trainer.model.predict(window_data.drop(columns=[target_column]))
        predictions.append(window_predictions[-1])  # Appending the prediction for the last day in the window

    for symbol, data in downloaded_data.items():
        visualizer.plot_data(data, title=f"Stock Price - {symbol}")

    # Plot real data and model predictions together
    visualizer.plot_predictions(new_data_df, predictions, title="Real vs Predicted Prices")


if __name__ == "__main__":
    main()
