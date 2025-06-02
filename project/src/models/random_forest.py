from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import yaml

from src.config.schemas import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, LoggingConfig
from src.data.dataset import TabularDataset

class RandomForestModel:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = RandomForestClassifier()
        self.data = TabularDataset(window_size=config.data.window_size)

    def load_data(self) -> pd.DataFrame:
        data_dict = self.data.load_data()
        return pd.concat([data_dict['numeric_data'], data_dict['categorical_data']], axis=1)

    def train(self):
        df = self.load_data()
        X = df.drop(columns=['target'])  # Assuming 'target' is the label column
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
