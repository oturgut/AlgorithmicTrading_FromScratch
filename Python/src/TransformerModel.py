import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from AbstractModel import AbstractModel


class TransformerModel(AbstractModel, TransformerMixin):
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.model = make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor()
        )

    def process(self, X, y, rolling_window_size):
        if rolling_window_size:
            self.window_size = rolling_window_size
        self.train(X, y)

    def transform(self, X, y):
        # Feature engineering: create rolling windows
        X_transformed = pd.DataFrame(index=X.index)
        X_transformed['lag'] = y.shift(self.window_size)
        # Include the other features from X
        for column in X.columns:
            X_transformed[column] = X[column]

        # Drop rows with NaN values
        X_transformed.dropna(inplace=True)

        return X_transformed

    def train(self, X, y):
        X_transformed = self.transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed.drop(columns=['lag']),
                                                            X_transformed['lag'], test_size=0.2, shuffle=False)

        self.model.fit(X_train, y_train)

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.model.predict(X_transformed)
