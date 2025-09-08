from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with the given configurations"""

        return CatBoostClassifier(**self.config, verbose=0)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, y):
        self.model.predict(X, y)

    def predict_proba(self, X):
        self.model.predict_proba(X)
