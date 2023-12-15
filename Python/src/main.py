
from datetime import datetime
from DataDownloader import DataDownloader
from Visualizer import Visualizer
from LinearRegressionModel import LinearRegressionModel
from ModelTrainer import ModelTrainer

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

    # Choose whether to retrain the model or use a pre-trained one
    trainer = ModelTrainer(model, retrain=True)

    # Assume data_df and target_column are available from your data
    # You can replace this with your actual data loading and preprocessing
    data_df = downloaded_data[symbols[0]]
    target_column = 'Adj Close'

    # Train the model
    trainer = ModelTrainer(model, data_df=data_df, target_column=target_column, retrain=True)

    # Save the model
    trainer.save_model(model_path)

    # Or load a pre-trained model
    # trainer = ModelTrainer(model, retrain=False)
    # trainer.load_model('your_pretrained_model_path')

    # Make predictions
    # predictions = trainer.model.predict(your_input_data)

    # Use the trained model to make predictions on another stock (e.g., MSFT)
    new_data_df = downloaded_data["MSFT"]
    print(new_data_df.head())
    predictions = trainer.model.predict(new_data_df.drop(columns=[target_column]))

    visualizer = Visualizer()
    for symbol, data in downloaded_data.items():
        visualizer.plot_data(data, title=f"Stock Price - {symbol}")

    # Plot real data and model predictions together
    visualizer.plot_predictions(new_data_df, predictions, title="Real vs Predicted Prices")

if __name__ == "__main__":
    main()
