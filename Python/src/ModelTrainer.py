import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
from AbstractModel import AbstractModel

class ModelTrainer:
    def __init__(self, model: AbstractModel, data_df=None, target_column=None, retrain=True, rolling_window_size=20):
        self.model = model
        self.rolling_window_size = rolling_window_size

        if retrain and data_df is not None and target_column is not None:
            self.train_model(data_df, target_column)

    def train_model(self, data_df: pd.DataFrame, target_column: str):
        X = data_df.drop(columns=[target_column])
        y = data_df[target_column]

        if self.rolling_window_size:
            # Use rolling window approach
            mse_values = []  # To store the MSE values for each prediction

            for i in range(len(X) - self.rolling_window_size):
                X_window = X.iloc[i:i + self.rolling_window_size]
                y_window = y.iloc[i:i + self.rolling_window_size]

                # Make sure X_window and y_window have the same number of rows
                min_rows = min(len(X_window), len(y_window))
                X_window = X_window.iloc[:min_rows]
                y_window = y_window.iloc[:min_rows]

                # Train the model on the current window using X_train and only the last point of y_train
                self.model.train(X_window, y_window)

                # Make predictions for the next data point
                X_next = X.iloc[i + 1:i + self.rolling_window_size + 1]
                y_pred = self.model.predict(X_next)

                # Evaluate the model on the current prediction and store the MSE
                mse = mean_squared_error(y.iloc[i + 1:i + self.rolling_window_size + 1], y_pred)
                mse_values.append(mse)
                print(f"MSE for prediction {i + self.rolling_window_size}: {mse}")

            # Calculate and print the aggregate MSE for the rolling window approach
            aggregate_mse = sum(mse_values) / len(mse_values)
            print(f"Aggregate Mean Squared Error (Rolling Window): {aggregate_mse}")

        else:
            # Use time series cross-validation
            cv = TimeSeriesSplit(n_splits=5)

            mse_values = []  # To store the MSE values for each fold

            for train_index, test_index in cv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train the model on the current split
                self.model.train(X_train, y_train)

                # Make predictions for the test set
                y_pred = self.model.predict(X_test)

                # Evaluate the model on the current fold and store the MSE
                mse = mean_squared_error(y_test, y_pred)
                mse_values.append(mse)
                print(f"Mean Squared Error (Fold {len(mse_values)}): {mse}")

            # Calculate and print the aggregate MSE for the time series cross-validation
            aggregate_mse = sum(mse_values) / len(mse_values)
            print(f"Aggregate Mean Squared Error (Time Series CV): {aggregate_mse}")

    def save_model(self, model_path: str):
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
