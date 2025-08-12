from xgboost import XGBClassifier

class XGBoostModel:

    def __init__(self, config):
        self.model = XGBClassifier(**config)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)