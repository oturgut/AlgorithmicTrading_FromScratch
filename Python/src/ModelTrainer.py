import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from AbstractModel import AbstractModel

class ModelTrainer:
    def __init__(self, model: AbstractModel, data_df=None, target_column=None, retrain=True):
        self.model = model

        if retrain and data_df is not None and target_column is not None:
            self.train_model(data_df, target_column)

    def train_model(self, data_df: pd.DataFrame, target_column: str):
        X = data_df.drop(columns=[target_column])
        y = data_df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        self.model.train(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def save_model(self, model_path: str):
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
