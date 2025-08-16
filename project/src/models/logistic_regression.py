from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Logistic Regression model with the given config."""
        return LogisticRegression(
            penalty=self.config.get("penalty"),
            C=self.config.get("C"),
            solver=self.config.get("solver"),
            max_iter=self.config.get("max_iter"),
            random_state=self.config.get("random_state"),
            l1_ratio=self.config.get("l1_ratio", None),  # safe default
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
