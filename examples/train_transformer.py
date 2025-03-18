import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any, Tuple, Optional, Union

from backend.app.utils.trackers import TransformerTracker  # Using LSTM tracker for Transformer as they're similar
from backend.app.utils.logger import Logger
from backend.app.pipelines.transformer_pipeline import TransformerPipeline
from backend.app.core.trainer import ModelTrainer

# Initialize logger
logger = Logger()


def initialize_tracking(config: Dict[str, Any], run_timestamp: str) -> TransformerTracker:
    """Initialize MLflow tracking."""
    experiment_name = config['mlflow']['experiment_name']
    run_name = f"transformer_{run_timestamp}"
    
    tracker = TransformerTracker(
        experiment_name=experiment_name,
        run_name=run_name
    )
    return tracker

if __name__ == "__main__":
    # Print usage instructions
    print("Training Transformer forecasting model")
    print("Usage: python train_transformer.py [--config CONFIG_PATH]")
    print("Example: python train_transformer.py --config config/config_transformer.json")
    print("\nStarting training process...\n")
    
    # Parse command line arguments
    trainer = ModelTrainer()
    args = trainer.parse_arguments()
    config_path = args.config
    
    # Load configuration and initialize tracking
    config = trainer.load_config(config_path)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tracker = initialize_tracking(config, run_timestamp)
    pipeline = TransformerPipeline(config=config, tracker=tracker)

    trainer.main(pipeline)
    tracker.end_run() 