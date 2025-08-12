from xgboost import XGBClassifier

class XGBoostModel:

    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()
    
    def _initalize_model(self):
        """Initialize the model with the given configurations"""
        return XGBClassifier(
            learning_rate = self.config.get("learning_rate"),
            max_depth = self.config.get("max_depth"),
            n_estimators = self.config.get("n_estimators"),
            scale_pos_weight = self.config.get("scale_pos_weight"),
            objective = self.config.get("objective"),
            eval_metric = self.config.get("eval_metric"),
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)