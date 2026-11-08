"""Leakage check: does any feature use future transactions or fraud labels?

Run (from this folder, with the virtual environment active):
    python check_leakage.py          # ~2-3 minutes; exit code 1 if the check fails

Method: build every feature twice,
  A) on the full data, as train.py does, and
  B) on the data cut off at 2019-09-01, with the fraud labels randomly shuffled,
then compare the values for the transactions before the cut-off. A feature that uses only earlier transactions
and no labels is unaffected by both changes, so its values come out identical. A feature that looks at later
transactions, or at labels, changes. Differences below 0.01% of a feature's spread count as floating-point
rounding.

Expected result:
  - all 60 history features: identical (no look-ahead, no labels)
  - the 3 job buying-power features: change, by design. They are fitted on the whole training period, which
    is look-ahead for training rows, but they use no fraud labels (see README)
  - the 8 deliberately leaky merchant features from leaky_reference.py: must change. They are the positive
    control; if they did not, the check itself would be broken
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import assign_partitions, prepare, read_transactions, require_files
from features import (BASE_NUM, HISTORY_FEATURES, JOB_F, MLOC_F, MP_F, SIM_F, TRAVEL_F, TSL_F, ULOC_F, UM_F, USER_F,
                      JobBuyingPower, add_history_features)
from leaky_reference import LEAKY_FEATURES, add_leaky_features

CUTOFF = '2019-09-01'
GROUPS = [('base', BASE_NUM), ('user history', USER_F), ('user x merchant', UM_F), ('merchant profiles', MP_F),
          ('job buying power', JOB_F), ('merchant similarity', SIM_F), ('merchant location', MLOC_F),
          ('user location', ULOC_F), ('travel', TRAVEL_F), ('time since last', TSL_F),
          ('leaky control (leaky_reference.py)', LEAKY_FEATURES)]


def build(df):
    """All features as train.py builds them, plus the deliberately leaky control features."""
    feats = add_history_features(df)
    assign_partitions(feats, config.VAL_FRAC)
    JobBuyingPower().fit(feats[feats.part == 'fit']).transform(feats)
    return add_leaky_features(feats, (feats.part == 'fit').to_numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=config.DATA_DIR, help='folder with fraudTrain.csv / fraudTest.csv')
    args = ap.parse_args()
    require_files(args.data_dir / config.TRAIN_FILE, args.data_dir / config.TEST_FILE)
    raw = pd.concat([read_transactions(args.data_dir / config.TRAIN_FILE, 'train'),
                     read_transactions(args.data_dir / config.TEST_FILE, 'test')], ignore_index=True)
    print('building features on the full data ...', flush=True)
    full = build(prepare(raw))
    print(f'building features on data before {CUTOFF} with shuffled labels ...', flush=True)
    cut = raw[pd.to_datetime(raw.trans_date_trans_time) < CUTOFF].copy()
    cut['is_fraud'] = np.random.RandomState(0).permutation(cut.is_fraud.to_numpy())
    part = build(prepare(cut))

    columns = [c for _, cols in GROUPS for c in cols]
    a = full.set_index('trans_num').loc[part.trans_num, columns].to_numpy(float)
    b = part[columns].to_numpy(float)
    tolerance = 1e-4 * np.nanstd(a, axis=0) + 1e-6
    changed = (~np.isclose(a, b, rtol=0, atol=tolerance, equal_nan=True)).sum(0)
    changed = pd.Series(changed, index=columns)

    print(f'\ncompared {len(part):,} transactions before {CUTOFF}\n')
    print(f'{"feature group":<34s} {"features":>8s} {"unchanged":>9s}   result')
    for label, cols in GROUPS:
        n_ok = int((changed[cols] == 0).sum())
        if cols is LEAKY_FEATURES:
            verdict = 'flagged as leaky (expected)' if n_ok == 0 else 'NOT FLAGGED - the check is broken'
        elif cols is JOB_F:
            verdict = 'look-ahead within training period (expected, no labels)' if n_ok < len(cols) else 'ok'
        else:
            verdict = 'ok' if n_ok == len(cols) else 'LEAK: ' + ', '.join(c for c in cols if changed[c])
        print(f'{label:<34s} {len(cols):>8d} {n_ok:>9d}   {verdict}')

    leaks = [c for c in HISTORY_FEATURES if changed[c]]
    control_ok = bool((changed[LEAKY_FEATURES] > 0).all())
    print(f'\nhistory features that use future data or labels: {leaks or "none"}')
    print(f'positive control (leaky_reference.py) flagged: {"yes" if control_ok else "NO"}')
    print('RESULT:', 'PASS' if not leaks and control_ok else 'FAIL')
    sys.exit(0 if not leaks and control_ok else 1)


if __name__ == '__main__':
    main()
