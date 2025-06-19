from sklearn.metrics import classification_report, accuracy_score

# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
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

    def run_training(self):
        try:
            # Data loading
            logger.info("Loading the data...")
            X, y = self.dataset.load_data()
            # split data into training and test sets
            # Using GroupShuffleSplit to ensure that the same group is not in both train and test sets
            logger.info("Splitting the data into training and test sets...")
            if X.empty or y.empty:
                raise ValueError("Loaded data is empty. Please check the dataset.")
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.data.test_size,
                random_state=self.config.random_state,
            )
            train_idx, test_idx = next(gss.split(X, y, groups=X["subject_id"]))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Training
            logger.info("Training the Random Forest model...")
            if X_train.empty or y_train.empty:
                raise ValueError("Training data is empty. Please check the dataset.")
            self.model.fit(X_train, y_train)

            # Evaluation
            logger.info("Evaluating the model...")
            y_pred = self.model.predict(X_test)
            # zero_division=0 to avoid division by zero in classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # logging model parameters
            logger.info("Logging model parameters...")
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
