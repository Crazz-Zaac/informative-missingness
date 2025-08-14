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
            scale_pos_weight=self.config.get("scale_pos_weight"),
            objective=self.config.get("objective"),
            eval_metric=self.config.get("eval_metric"),
            task_type='GPU',
            devices='0'
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X, y):
        self.model.predict(X, y)
    
    def predict_proba(self, X):
        self.model.predict_proba(X)