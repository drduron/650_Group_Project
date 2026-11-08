"""Reading, validating and cleaning the raw transaction files.

The Sparkov dataset ships as CSV files with one row per card transaction. `read_transactions` reads one file
and checks its columns; `prepare` converts types, adds an integer timestamp and sorts the rows by card and
time, which is the order every history feature relies on. `assign_partitions` creates the time-based
fit / validation / test split used for training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL = 'is_fraud'
CATEGORICAL = ['merchant', 'category', 'gender', 'state', 'job', 'city']
NUMERIC = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
REQUIRED_COLUMNS = ['trans_date_trans_time', 'trans_num', 'dob'] + NUMERIC + CATEGORICAL


def require_files(*paths):
    """Exit with a helpful message if any input file is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(f'missing data file(s): {missing}\nRun `python download_data.py` first, or point --data-dir / '
                 f'--input / --history at the files.')


def read_transactions(path, source):
    """Read one transaction CSV and tag every row with `source` (e.g. 'train', 'test', 'history', 'input').

    The label column is optional, so unlabelled transactions can be scored.
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])   # the CSVs carry a saved row index
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}')
    df['source'] = source
    return df


def prepare(raw):
    """Clean a raw transaction table: numeric and datetime types, integer epoch seconds ('ts'), string
    categoricals, and sort by (card, time, transaction id). Raises if any value cannot be parsed."""
    df = raw.reset_index(drop=True)
    if df.trans_num.duplicated().any():
        raise ValueError(f'{int(df.trans_num.duplicated().sum())} duplicated trans_num values')
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[LABEL] = pd.to_numeric(df[LABEL], errors='coerce') if LABEL in df.columns else np.nan
    df['dt'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    bad = df[NUMERIC + ['dt', 'dob']].isna().sum()
    if bad.any():
        raise ValueError(f'unparseable values: {bad[bad > 0].to_dict()}')
    # integer epoch seconds, used for fast time-window arithmetic in the history features
    df['ts'] = ((df['dt'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')).astype('int64')
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
    return df.sort_values(['cc_num', 'ts', 'trans_num']).reset_index(drop=True)


def assign_partitions(df, val_frac):
    """Time-based split used for training.

    test = rows from the test file; the training file is cut at its (1 - val_frac) time quantile into
    'fit' (earlier, used to train) and 'val' (later, used for early stopping and the decision threshold).
    """
    is_train = (df.source == 'train').to_numpy()
    cut = np.quantile(df.ts[is_train], 1 - val_frac)
    df['part'] = np.where(~is_train, 'test', np.where(df.ts <= cut, 'fit', 'val'))
    return df
