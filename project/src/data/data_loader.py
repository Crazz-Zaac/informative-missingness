from config import DataConfig
from typing import Optional, List
from pathlib import Path


class DataLoaderConfig:
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def preprocess_data(self):
        # Placeholder for data preprocessing logic
        pass

    def load_data(self):
        # Placeholder for data loading logic
        pass

    def __repr__(self):
        return f"DataLoaderConfig(data_dir={self.data_dir}, batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle})"
