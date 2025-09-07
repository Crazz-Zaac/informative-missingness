from xgboost import XGBClassifier


class XGBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with the given configurations"""
        return XGBClassifier(**self.config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
