from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the Gradient Boosting model with the given configuration."""
        return GradientBoostingClassifier(
            learning_rate=self.config.get("learning_rate"),
            max_depth=self.config.get("max_depth"),
            n_estimators=self.config.get("n_estimators"),
            min_samples_split=self.config.get("min_samples_split"),
            min_samples_leaf=self.config.get("min_samples_leaf"),
        )

    def fit(self, X, y):
        """Fit the Gradient Boosting model to the training data."""
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)