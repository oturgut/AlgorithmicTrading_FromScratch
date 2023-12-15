class AbstractModel:
    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement the train method.")

    def predict(self, X):
        raise NotImplementedError("Subclasses must implement the predict method.")
