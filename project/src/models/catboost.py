from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the model with the given configurations"""

        return CatBoostClassifier(
            learning_rate=self.config.get("learning_rate"),
            depth=self.config.get("depth"),
            iterations=self.config.get("iterations"),
            loss_function=self.config.get("loss_function"),
            l2_leaf_reg=self.config.get("l2_leaf_reg"),
            border_count=self.config.get("border_count"),
            eval_metric=self.config.get("eval_metric"),
            task_type="GPU",
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, y):
        self.model.predict(X, y)

    def predict_proba(self, X):
        self.model.predict_proba(X)
