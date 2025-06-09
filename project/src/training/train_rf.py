from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import yaml
from src.config.schemas import ExperimentConfig
from src.data.dataset import TabularDataset
from src.models.random_forest import RandomForestModel
from loguru import logger


class RandomForestTrainer:
    """Handles the complete training process"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        # self.logger = ExperimentLogger(config)
        self.dataset = TabularDataset(
            window_size=self.config.data.tabular.window_size, config=self.config
        )
        self.model = RandomForestModel(config=config.model)

    def load_data(self) -> tuple:
        df = self.dataset.load_data()
        return df.drop(columns=["target"]), df["target"]

    def run_training(self):
        try:
            # Data loading
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.data.test_size,
                random_state=self.config.random_state,
            )

            # Training
            self.model.fit(X_train, y_train)

            # Evaluation
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            # logging model parameters
            logger.info("Experiment started...")
            for key, value in self.config.model.hyperparameters.model_dump().items():
                logger.info(f"Parameter - {key}: {value}")
            
            # Logging
            logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            logger.info(f"F1 Score: {report['weighted avg']['f1-score']:.4f}")

            return self.model

        finally:
            logger.info("Experiment completed.")

    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = ExperimentConfig(**yaml.safe_load(f))
        return cls(config)
