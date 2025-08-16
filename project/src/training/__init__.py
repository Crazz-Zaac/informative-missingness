from src.training.train_rf import RandomForestTrainer
from src.training.train_gradboost import GradientBoostingTrainer
# from src.training.train_logreg import LogisticRegressionTrainer
from src.training.train_xgboost import XGBoostTrainer
from src.training.train_catboost import CatBoostTrainer

TRAINER_REGISTRY = {
    "randomforest": RandomForestTrainer,
    "gradientboosting": GradientBoostingTrainer,
    # "logisticregression": LogisticRegressionTrainer,
    "xgboost": XGBoostTrainer,
    "catboost": CatBoostTrainer
}
