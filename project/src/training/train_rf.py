from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import yaml
from src.config.schemas import ExperimentConfig
from src.data.dataset import TabularDataset
from src.models.random_forest import RandomForestModel
from src.training.logger import ExperimentLogger

class RandomForestTrainer:
    """Handles the complete training process"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = ExperimentLogger(config)
        self.dataset = TabularDataset(window_size=config.data.window_size)
        self.model = RandomForestModel(config.model)

    def load_data(self) -> tuple:
        data_dict = self.dataset.load_data()
        df = pd.concat([data_dict["numeric_data"], data_dict["categorical_data"]], axis=1)
        return df.drop(columns=["target"]), df["target"]

    def run_training(self):
        try:
            # Data loading
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.data.test_size,
                random_state=self.config.random_state
            )

            # Training
            self.model.model.fit(X_train, y_train)

            # Evaluation
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Logging
            self.logger.log_metrics({
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': report['weighted avg']['f1-score']
            })
            
            return self.model
            
        finally:
            self.logger.finish()

    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = ExperimentConfig(**yaml.safe_load(f))
        return cls(config)