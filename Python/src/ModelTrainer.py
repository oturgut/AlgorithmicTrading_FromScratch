import pandas as pd
import numpy as np
import joblib
from AbstractModel import AbstractModel

class ModelTrainer:
    def __init__(self, model: AbstractModel, data_df=None, target_column=None, retrain=True, rolling_window_size=20):
        self.model = model
        self.rolling_window_size = rolling_window_size
        if hasattr(model, 'window_size'):
            self.rolling_window_size = model.window_size

        if retrain and data_df is not None and target_column is not None:
            self.train_model(data_df, target_column)

    def train_model(self, data_df: pd.DataFrame, target_column: str):
        X = data_df.drop(columns=[target_column])
        y = data_df[target_column]
        # Which calls processing the input and the training.
        self.model.process(X, y, self.rolling_window_size)

    def save_model(self, model_path: str):
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
