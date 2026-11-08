"""Project-wide settings: file locations, data split and random seed.

Every path is relative to this folder, so the project runs from wherever it is cloned. Scripts also take
--data-dir / --artifacts-dir / --results-dir arguments to override these defaults.
"""
import os
from pathlib import Path

PROJECT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

DATA_DIR = PROJECT_DIR / 'data'            # raw CSVs (downloaded by download_data.py)
ARTIFACTS_DIR = PROJECT_DIR / 'artifacts'  # saved models + training-period state (train.py writes, predict.py reads)
RESULTS_DIR = PROJECT_DIR / 'results'      # metrics, feature importances, predictions, logs

TRAIN_FILE = 'fraudTrain.csv'              # 2019-01-01 .. 2020-06-21, 1,296,675 transactions
TEST_FILE = 'fraudTest.csv'                # 2020-06-21 .. 2020-12-31,   555,719 transactions

VAL_FRAC = 0.15                            # the last 15% of the training file (by time) is the validation set
SEED = 42                                  # random seed for all models


def display_path(path):
    """Path for log messages: relative to the project folder when inside it, so logs contain no machine paths.
    Symlinks are deliberately not resolved (e.g. data/ may be a symbolic link to a folder elsewhere)."""
    path = Path(os.path.abspath(path))
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)
