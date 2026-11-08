"""Build the feature pipeline, train CatBoost / XGBoost / LightGBM, evaluate them on validation and test,
and save everything predict.py needs.

Usage (from this folder, with the virtual environment active):
    python train.py                                   # all three models (~4-10 min on a laptop)
    python train.py --models xgboost lightgbm         # a subset
    python train.py --seed 1 --artifacts-dir runs/seed1/artifacts --results-dir runs/seed1/results

Steps:
  1. Load and clean data/fraudTrain.csv and data/fraudTest.csv
  2. Compute the history features over the whole timeline (each row only sees earlier transactions)
  3. Split by time: fit = first 85% of the training file, val = last 15%, test = the test file
  4. Fit the job buying-power model on the fit rows and apply it to every row
  5. For each model: train on fit, early-stop on validation PR-AUC, pick the F1-maximising threshold on
     validation, then evaluate on validation and test with that threshold
  6. Save each model, and mark the one with the highest VALIDATION PR-AUC as the best model
     (the test set is only reported, never used for any choice)

Outputs:
  artifacts/<model>/                model file, encoder, feature lists, threshold (model_info.json)
  artifacts/job_buying_power.pkl    fitted JobBuyingPower (needed to build features for new rows)
  artifacts/metadata.json           best model, metrics, data periods, feature lists, library versions
  results/metrics.csv               one row per model: validation and test metrics
  results/feature_importance/<model>.csv
  results/test_predictions/<model>.csv   fraud probability for every test transaction
  results/train.log                 this run's log
"""
import argparse
import json
import logging
import pickle
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import config
from data import LABEL, assign_partitions, prepare, read_transactions, require_files
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, JobBuyingPower, add_history_features
from models import MODELS, best_f1_threshold, evaluate, load_model

log = logging.getLogger('fraudwatch')


def setup_logging(log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s  %(message)s', '%H:%M:%S')
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='w')):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def build_feature_table(data_dir):
    """Steps 1-4: load, clean, compute features, split, and return (feature table, fitted JobBuyingPower)."""
    t0 = time.time()
    require_files(data_dir / config.TRAIN_FILE, data_dir / config.TEST_FILE)
    raw = pd.concat([read_transactions(data_dir / config.TRAIN_FILE, 'train'),
                     read_transactions(data_dir / config.TEST_FILE, 'test')], ignore_index=True)
    df = prepare(raw)
    del raw
    if df[LABEL].isna().any():
        raise ValueError('training and test files must contain the is_fraud label')
    log.info(f'loaded {len(df):,} transactions from {config.display_path(data_dir)} ({time.time() - t0:.0f}s)')

    t0 = time.time()
    feats = add_history_features(df, log=log.info)
    del df                                           # intermediate columns are no longer needed
    assign_partitions(feats, config.VAL_FRAC)
    job = JobBuyingPower().fit(feats[feats.part == 'fit'])
    job.transform(feats)
    log.info(f'features ready: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical ({time.time() - t0:.0f}s)')
    return feats, job


def describe_split(feats):
    """Rows, fraud count and date range of each partition (logged and stored in metadata.json)."""
    info = {}
    for part in ('fit', 'val', 'test'):
        rows = feats[feats.part == part]
        info[part] = dict(rows=len(rows), fraud=int(rows[LABEL].sum()), fraud_rate=round(float(rows[LABEL].mean()), 5),
                          fraud_cards=int(rows.loc[rows[LABEL] == 1, 'cc_num'].nunique()),
                          start=rows.trans_date_trans_time.min(), end=rows.trans_date_trans_time.max())
        log.info(f'  {part:<4s} {info[part]["rows"]:>9,} rows | {info[part]["fraud"]:>5,} fraud ({info[part]["fraud_rate"]:.2%}) '
                 f'on {info[part]["fraud_cards"]} cards | {info[part]["start"]} -> {info[part]["end"]}')
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS), help='models to train')
    ap.add_argument('--seed', type=int, default=config.SEED, help='random seed for the models')
    ap.add_argument('--data-dir', type=Path, default=config.DATA_DIR, help='folder with fraudTrain.csv / fraudTest.csv')
    ap.add_argument('--artifacts-dir', type=Path, default=config.ARTIFACTS_DIR, help='where models are saved')
    ap.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR, help='where metrics and logs are written')
    args = ap.parse_args()
    setup_logging(args.results_dir / 'train.log')
    for sub in ('feature_importance', 'test_predictions'):
        (args.results_dir / sub).mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    feats, job = build_feature_table(args.data_dir)
    split = describe_split(feats)
    with open(args.artifacts_dir / 'job_buying_power.pkl', 'wb') as f:
        pickle.dump(job, f)
    fit, val, test = (feats[feats.part == p] for p in ('fit', 'val', 'test'))

    rows = []
    for name in args.models:
        log.info(f'training {name} (seed {args.seed})')
        t0 = time.time()
        model = MODELS[name](NUMERIC_FEATURES, CATEGORICAL_FEATURES, seed=args.seed).fit(fit, val)
        secs = time.time() - t0
        # Evaluate the saved model file, exactly as predict.py will load it: save, reload, then predict.
        model.save(args.artifacts_dir / name)
        model = load_model(args.artifacts_dir / name)
        p_val, p_test = model.predict_proba(val), model.predict_proba(test)
        model.threshold = best_f1_threshold(val[LABEL], p_val)
        m_val, m_test = evaluate(val[LABEL], p_val, model.threshold), evaluate(test[LABEL], p_test, model.threshold)
        model.save(args.artifacts_dir / name)        # again, now including the decision threshold

        rows.append(dict(model=name, **{f'val_{k}': v for k, v in m_val.items() if k != 'threshold'},
                         **{f'test_{k}': v for k, v in m_test.items()}, rounds=model.best_iteration, train_secs=round(secs)))
        log.info(f'  {name}: val PR-AUC {m_val["pr_auc"]:.4f} | test PR-AUC {m_test["pr_auc"]:.4f} | test precision '
                 f'{m_test["precision"]:.3f} recall {m_test["recall"]:.3f} F1 {m_test["f1"]:.3f} @ threshold '
                 f'{model.threshold:.3f} | {model.best_iteration} rounds | {secs:.0f}s')
        imp = model.feature_importance()
        (100 * imp / imp.sum()).sort_values(ascending=False).rename('importance_pct').to_csv(
            args.results_dir / 'feature_importance' / f'{name}.csv', index_label='feature')
        pd.DataFrame({'trans_num': test.trans_num.values, LABEL: test[LABEL].astype(int).values,
                      'fraud_probability': p_test}).to_csv(args.results_dir / 'test_predictions' / f'{name}.csv', index=False)

    # metadata.json keeps one entry per saved model. Models not retrained in this run keep their earlier entry,
    # so `python train.py --models xgboost` still compares against previously trained CatBoost / LightGBM.
    meta_path = args.artifacts_dir / 'metadata.json'
    models_meta = json.loads(meta_path.read_text())['models'] if meta_path.exists() else {}
    now = datetime.now().isoformat(timespec='seconds')
    for r in rows:
        models_meta[r['model']] = dict(trained=now, seed=args.seed, rounds=int(r['rounds']), train_secs=int(r['train_secs']),
                                       val={k[4:]: float(v) for k, v in r.items() if k.startswith('val_')},
                                       test={k[5:]: float(v) for k, v in r.items() if k.startswith('test_')})
    models_meta = {n: m for n, m in models_meta.items() if (args.artifacts_dir / n / 'model_info.json').exists()}
    best = max(models_meta, key=lambda n: models_meta[n]['val']['pr_auc'])   # chosen on validation, never on test

    meta = dict(updated=now, best_model=best, best_model_rule='highest validation PR-AUC', models=models_meta,
                split=split, numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES,
                versions=dict(python=platform.python_version(),
                              **{p: version(p) for p in ('pandas', 'numpy', 'scikit-learn', 'catboost', 'xgboost', 'lightgbm')}))
    meta_path.write_text(json.dumps(meta, indent=2))

    # results/metrics.csv: the same information as a table, one row per saved model
    table = pd.DataFrame({n: {**{f'val_{k}': v for k, v in m['val'].items()}, **{f'test_{k}': v for k, v in m['test'].items()},
                              'rounds': m['rounds'], 'train_secs': m['train_secs'], 'seed': m['seed'], 'trained': m['trained']}
                          for n, m in models_meta.items()}).T.rename_axis('model')
    out = args.results_dir / 'metrics.csv'
    table.to_csv(out)
    cols = ['val_pr_auc', 'test_pr_auc', 'test_auc_roc', 'test_precision', 'test_recall', 'test_f1', 'test_threshold', 'rounds']
    show = table[cols].astype(float).astype({'rounds': int})
    log.info('\n' + show.to_string(float_format='{:.4f}'.format) + '\n')
    log.info(f'best model (highest validation PR-AUC): {best}')
    log.info(f'saved models to {config.display_path(args.artifacts_dir)}/, metrics to {config.display_path(out)}')


if __name__ == '__main__':
    main()
