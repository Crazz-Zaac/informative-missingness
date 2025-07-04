from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    accuracy_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import datetime
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
        self.random_state = 42  # config.model.hyperparameters.random_state

    def save_and_plot(
        self, mean_importance: pd.DataFrame, recalls: list, f1s: list, aucs: list
    ):
        parent_dir = Path(__file__).parent.parent.parent
        plot_dir = parent_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        mean_importance.plot(
            kind="barh", figsize=(10, 6), title="Permutation Feature Importance"
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/feature_importance_{timestamp}.png")

        metrics_df = pd.DataFrame(
            {"fold": list(range(1, 6)), "recall": recalls, "f1": f1s, "roc_auc": aucs}
        )
        metrics_df.to_csv(f"{plot_dir}/fold_metrics_{timestamp}.csv", index=False)

    def run_training(self):
        recalls, f1s, aucs = [], [], []
        all_importances = []
        logger.info("Loading and preparing the data...")
        X, y = self.dataset.load_and_split_data()
        groups = X["subject_id"]
        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )
        splits = sgkf.split(X, y, groups=groups)
        try:
            for fold, (train_idx, test_idx) in enumerate(splits):
                logger.info(f"Running fold {fold+1}...")

                X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                # checking column types
                X_train.columns = X_train.columns.astype(str)
                X_test.columns = X_test.columns.astype(str)

                logger.info("Imputing data using KNN Imputer")
                imputer = KNNImputer(n_neighbors=5)
                X_train_imputed = pd.DataFrame(
                    imputer.fit_transform(X_train), columns=X_train.columns
                )
                X_test_imputed = pd.DataFrame(
                    imputer.transform(X_test), columns=X_test.columns
                )

                logger.info("Training the Random Forest model...")
                model = RandomForestModel(config=self.config.model)
                if X_train.empty or y_train.empty:
                    raise ValueError(
                        "Training data is empty. Please check the dataset."
                    )
                model.fit(X_train_imputed, y_train)

                logger.info("Evaluating the model...")
                y_pred = model.predict(X_test_imputed)
                y_prob = model.predict_proba(X_test_imputed)[:, 1]
                # report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                recalls.append(recall_score(y_test, y_pred, pos_label=1))
                f1s.append(f1_score(y_test, y_pred, pos_label=1))
                aucs.append(roc_auc_score(y_test, y_prob))

                result = permutation_importance(
                    self.model.model,  # access underlying sklearn model
                    X_test,
                    y_test,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1,
                )
                all_importances.append(
                    pd.Series(result.importances_mean, index=X_test.columns)
                )

            # Compute permutation importance
            logger.info("Computing permutation feature importances...")
            mean_importance = (
                pd.concat(all_importances, axis=1)
                .mean(axis=1)
                .sort_values(ascending=False)
            )

            logger.info("Saving and plotting the important features")
            self.save_and_plot(
                mean_importance=mean_importance, recalls=recalls, f1s=f1s, aucs=aucs
            )

            # Log top features
            logger.info("Top 10 Permutation Feature Importances:")
            for feature, importance in mean_importance.head(10).items():
                logger.info(f"{feature}: {importance:.4f}")

            logger.info("Logging model parameters...")
            for key, value in self.config.model.hyperparameters.model_dump().items():
                logger.info(f"Parameter - {key}: {value}")

            logger.info(
                f"Mean Recall (class=1): {pd.Series(recalls).mean():.4f} ± {pd.Series(recalls).std():.4f}"
            )
            logger.info(
                f"Mean F1 (class=1): {pd.Series(f1s).mean():.4f} ± {pd.Series(f1s).std():.4f}"
            )
            logger.info(
                f"Mean ROC-AUC: {pd.Series(aucs).mean():.4f} ± {pd.Series(aucs).std():.4f}"
            )

            return self.model

        except Exception as e:
            logger.exception("Experiment failed due to an unexpected error.")

        finally:
            logger.success("Experiment finished.")

    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path, "r") as f:
            config = ExperimentConfig(**yaml.safe_load(f))
        return cls(config)
