from sklearn.ensemble import RandomForestClassifier
from src.config.schemas import RandomForestFixedParams


class RandomForestModel:
    def __init__(self, config: RandomForestFixedParams):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the Random Forest model with the given configuration."""
        return RandomForestClassifier(
            n_estimators=self.config.get("n_estimators"),
            max_depth=self.config.get("max_depth"),
            class_weight=self.config.get("class_weight"),
            random_state=self.config.get("random_state"),
            n_jobs=-1,
            min_samples_split=self.config.get("min_samples_split"),
            min_samples_leaf=self.config.get("min_samples_leaf"),
        )

    def fit(self, X, y):
        """Fit the Random Forest model to the training data."""
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
