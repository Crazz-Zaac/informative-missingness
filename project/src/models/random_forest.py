from sklearn.ensemble import RandomForestClassifier
from src.config.schemas import ModelConfig


class RandomForestModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the Random Forest model with the given configuration."""
        return RandomForestClassifier(
            n_estimators=self.config.hyperparameters.n_estimators,
            max_depth=self.config.hyperparameters.max_depth,
            class_weight=self.config.hyperparameters.class_weight,
            random_state=self.config.hyperparameters.random_state,
            n_jobs=-1,
            min_samples_split=self.config.hyperparameters.min_samples_split,
            min_samples_leaf=self.config.hyperparameters.min_samples_leaf,
        )
    
    def fit(self, X, y):
        """Fit the Random Forest model to the training data."""
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def feature_importances(self):
        return self.model.feature_importances_
