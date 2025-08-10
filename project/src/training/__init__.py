from src.training.train_rf import RandomForestTrainer
from src.training.train_gradboost import GradientBoostingTrainer
# from src.training.train_logreg import LogisticRegressionTrainer
# from src.training.train_xgb import XGBoostTrainer

TRAINER_REGISTRY = {
    "randomforest": RandomForestTrainer,
    "gradientboosting": GradientBoostingTrainer,
    # "logisticregression": LogisticRegressionTrainer,
    # "xgboost": XGBoostTrainer,
}
