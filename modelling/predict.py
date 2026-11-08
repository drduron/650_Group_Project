"""Score transactions with a saved fraud model (inference).

Usage (from this folder, with the virtual environment active, after train.py has run):
    python predict.py                          # score data/fraudTest.csv with the best saved model
    python predict.py --limit 1000             # only the first 1,000 test transactions (in time order)
    python predict.py --model xgboost          # use a specific saved model
    python predict.py --input new.csv --history data/fraudTrain.csv data/fraudTest.csv --output new_scores.csv

Why it needs history: the features describe each transaction relative to EARLIER transactions of the same card,
merchant and category (e.g. "amount vs this card's average over the last 3 days"). Scoring a transaction
therefore needs the transactions that came before it. By default the training file is the history for the test
file. For new data, pass every earlier transaction you have.

Steps:
  1. Read the rows to score (--input) and the history (--history). A transaction present in both counts as input.
  2. Build the same features as train.py: history features + the saved job buying-power model.
  3. Apply the saved model and its saved decision threshold to the input rows.
  4. Write one row per input transaction (input order): trans_num, cc_num, time, merchant, category, amount,
     fraud_probability, fraud_flag (1 = probability >= threshold), and is_fraud when the input has labels.
  5. If the input has is_fraud labels, print the metrics. If train.py's test predictions are available, also
     check that the probabilities match them (they should be identical for the test file).
"""
import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import LABEL, prepare, read_transactions, require_files
from features import add_history_features
from models import evaluate, load_model

log = logging.getLogger('fraudwatch')
TOLERANCE = 1e-6     # differences below this count as floating-point rounding (expected: exactly 0)


def load_rows(input_path, history_paths, limit=None):
    """Read input and history CSVs into one prepared table (column 'source' = 'input' / 'history')."""
    inp = read_transactions(input_path, 'input')
    if limit:
        order = pd.to_datetime(inp.trans_date_trans_time).sort_values(kind='stable').index
        inp = inp.loc[order[:limit]]
    frames = [read_transactions(p, 'history') for p in history_paths]
    hist = pd.concat(frames, ignore_index=True) if frames else inp.iloc[:0]
    hist = hist[~hist.trans_num.isin(inp.trans_num)]
    return inp.trans_num.tolist(), prepare(pd.concat([hist, inp], ignore_index=True))


def compare_with_training(out, model_name, results_dir):
    """Compare probabilities with those train.py saved for the test file (only for overlapping transactions)."""
    path = results_dir / 'test_predictions' / f'{model_name}.csv'
    if not path.exists():
        log.info(f'per-transaction check skipped: {config.display_path(path)} is written by train.py (not stored in git)')
        return
    ref = pd.read_csv(path).set_index('trans_num').fraud_probability
    common = out.trans_num[out.trans_num.isin(ref.index)]
    if len(common):
        diff = np.abs(out.set_index('trans_num').fraud_probability.loc[common] - ref.loc[common]).max()
        log.info(f'check vs training-time test predictions: {len(common):,} common transactions, '
                 f'max |probability difference| = {diff:.1e} -> {"IDENTICAL" if diff < TOLERANCE else "DIFFERENT"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=config.DATA_DIR / config.TEST_FILE, help='CSV of transactions to score')
    ap.add_argument('--history', type=Path, nargs='*', default=[config.DATA_DIR / config.TRAIN_FILE],
                    help='CSV(s) of earlier transactions used to build the features (default: the training file)')
    ap.add_argument('--model', choices=['catboost', 'xgboost', 'lightgbm'], help='saved model to use (default: best)')
    ap.add_argument('--limit', type=int, help='score only the first N input transactions in time order')
    ap.add_argument('--output', type=Path, help='output CSV (default: results/predictions_<model>.csv)')
    ap.add_argument('--artifacts-dir', type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)

    # 1. saved artifacts
    meta_path = args.artifacts_dir / 'metadata.json'
    if not meta_path.exists():
        sys.exit(f'no trained models in {config.display_path(args.artifacts_dir)} - run train.py first')
    meta = json.loads(meta_path.read_text())
    name = args.model or meta['best_model']
    available = sorted(n for n in meta['models'] if (args.artifacts_dir / n / 'model_info.json').exists())
    if name not in available:
        sys.exit(f'model {name!r} is not in {config.display_path(args.artifacts_dir)} (available: {available}). '
                 f'Create it with `python train.py --models {name}`.')
    require_files(args.input, *args.history)
    model = load_model(args.artifacts_dir / name)
    with open(args.artifacts_dir / 'job_buying_power.pkl', 'rb') as f:
        job = pickle.load(f)
    log.info(f'model: {name}{" (best by validation PR-AUC)" if name == meta["best_model"] else ""} | '
             f'threshold {model.threshold:.4f} | trained {meta["models"][name]["trained"]}')

    # 2. rows + history -> features
    t0 = time.time()
    input_ids, df = load_rows(args.input, args.history, args.limit)
    log.info(f'input: {len(input_ids):,} transactions from {args.input.name} | history: '
             f'{(df.source == "history").sum():,} earlier transactions from {[p.name for p in args.history]}')
    feats = add_history_features(df)
    del df
    job.transform(feats)                                   # on the full table: card attributes come from its first row
    feats = feats[feats.source == 'input']
    log.info(f'features built ({time.time() - t0:.0f}s)')

    # 3. score
    p = model.predict_proba(feats)
    out = feats[['trans_num', 'cc_num', 'trans_date_trans_time', 'merchant', 'category', 'amt', LABEL]].copy()
    out['fraud_probability'] = p
    out['fraud_flag'] = (p >= model.threshold).astype(int)
    out = out.set_index('trans_num').loc[input_ids].reset_index()            # back to input order
    labelled = out[LABEL].notna().all()
    if labelled:
        out[LABEL] = out[LABEL].astype(int)
    else:
        out = out.drop(columns=LABEL)

    # 4. write
    output = args.output or args.results_dir / f'predictions_{name}.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    log.info(f'scored {len(out):,} transactions: {out.fraud_flag.sum():,} flagged as fraud ({out.fraud_flag.mean():.2%}) '
             f'-> {config.display_path(output)}')
    preview = out.nlargest(5, 'fraud_probability')[['trans_num', 'trans_date_trans_time', 'category', 'amt', 'fraud_probability']]
    log.info(f'highest-scoring transactions:\n{preview.to_string(index=False)}')

    # 5. evaluation (labelled input only)
    if labelled and out[LABEL].nunique() < 2:
        log.info(f'all {len(out):,} input transactions have is_fraud = {out[LABEL].iloc[0]}: metrics are not defined')
    elif labelled:
        m = evaluate(out[LABEL], out.fraud_probability.to_numpy(), model.threshold)
        log.info(f'metrics on these {len(out):,} labelled transactions ({out[LABEL].sum():,} fraud): '
                 f'PR-AUC {m["pr_auc"]:.4f} | ROC-AUC {m["auc_roc"]:.4f} | precision {m["precision"]:.4f} | '
                 f'recall {m["recall"]:.4f} | F1 {m["f1"]:.4f}')
        ref = meta['models'][name]['test']
        if len(out) == meta['split']['test']['rows']:
            same = all(abs(m[k] - ref[k]) < TOLERANCE for k in ('pr_auc', 'auc_roc', 'precision', 'recall', 'f1'))
            log.info(f'training-time test metrics: PR-AUC {ref["pr_auc"]:.4f} | precision {ref["precision"]:.4f} | '
                     f'recall {ref["recall"]:.4f} | F1 {ref["f1"]:.4f} -> {"MATCH" if same else "DIFFERENT"}')
    if labelled:
        compare_with_training(out, name, args.results_dir)


if __name__ == '__main__':
    main()
