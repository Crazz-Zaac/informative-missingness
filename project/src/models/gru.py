from loguru import logger
from src.config.schemas import GRUModelParams
import torch.nn as nn


class GRUModelConfig:
    def __init__(self, config: GRUModelParams):
        self.config = config
        self.gru_model = self._initialize_model__()

    def _initialize_model(self):
        """Initialize the GRU model with the given configuration."""
        logger.info("Initializing GRU model with the provided configuration.")
        return nn.GRU(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            bidirectional=self.config.bidirectional,
        )
    
    
