from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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
            logger.info("Loading and preparing the data...")
            X_train, X_test, y_train, y_test = self.dataset.load_and_split_data()

            logger.info("Training the Random Forest model...")
            if X_train.empty or y_train.empty:
                raise ValueError("Training data is empty. Please check the dataset.")
            self.model.fit(X_train, y_train)

            logger.info("Evaluating the model...")
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            # Compute permutation importance
            logger.info("Computing permutation feature importances...")
            perm_importance = permutation_importance(
                self.model.model,  # access underlying sklearn model
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )

            # Convert to pandas Series for easy plotting/logging
            importances = pd.Series(perm_importance.importances_mean, index=X_test.columns)
            importances = importances.sort_values(ascending=False)

            # Log top features
            logger.info("Top 10 Permutation Feature Importances:")
            for feature, importance in importances.head(10).items():
                logger.info(f"{feature}: {importance:.4f}")

            # Plot the importances (optional)
            importances.plot(kind="barh", figsize=(10, 6), title="Permutation Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()


            logger.info("Logging model parameters...")
            for key, value in self.config.model.hyperparameters.model_dump().items():
                logger.info(f"Parameter - {key}: {value}")

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
