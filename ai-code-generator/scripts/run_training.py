# scripts/run_training.py

import argparse
import logging
from pathlib import Path
import sys

# Adiciona o diretório raiz do projeto ao PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.models.codet5_model import CodeT5Model, CodeT5Config
from src.data.data_loader import load_dataset
from src.utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train the CodeT5 model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-small", help="Name of the pre-trained model to use")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for training")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Loading dataset...")
    train_data, val_data = load_dataset(args.data_path)

    logger.info("Initializing model...")
    config = CodeT5Config(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    model = CodeT5Model(config)

    logger.info("Starting training...")
    model.train(train_data, val_data)

    logger.info(f"Saving model to {args.output_dir}")
    model.save_model(args.output_dir)

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
