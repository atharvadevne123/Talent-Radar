"""CLI script to train the Talent-Radar salary prediction model."""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train the Talent-Radar salary model")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of synthetic training samples")
    parser.add_argument("--model-path", type=str, default=None, help="Override MODEL_PATH env var")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def main() -> int:
    """Train the model and print metrics.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    try:
        from app.model import generate_synthetic_training_data, train
        logger.info("Generating %d training samples...", args.n_samples)
        X, y = generate_synthetic_training_data(n=args.n_samples)
        logger.info("Training model...")
        metrics = train(X, y)
        print(f"Training complete: CV MAE=${metrics['cv_mae']:,.0f}  CV R2={metrics['cv_r2']:.4f}")
        return 0
    except Exception:
        logger.exception("Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
