"""Models, training procedure and evaluation metrics.

Each model class wraps one gradient-boosting library behind the same interface:

    model = MODELS['xgboost'](numeric_features, categorical_features, seed=42)
    model.fit(fit_df, val_df)        # train on the fit rows, early-stop on validation PR-AUC
    p = model.predict_proba(df)      # fraud probability for each row of a feature table
    model.save(directory)            # model file + encoder + feature lists + decision threshold
    model = load_model(directory)

Training procedure (the same for all models): at most 600 boosting rounds; after every round the
validation PR-AUC (scikit-learn average precision) is computed; training stops when it has not improved for
50 rounds; the model is cut back to the best round. The decision threshold is chosen afterwards on validation
(best_f1_threshold) and applied unchanged to any later data.
"""
import json
import logging
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from data import LABEL

log = logging.getLogger('fraudwatch')
warnings.filterwarnings('ignore', message='Failed to import numba')   # CatBoost: optional speed-up, not needed

MAX_ITERS = 600     # maximum boosting rounds
PATIENCE = 50       # stop after this many rounds without a better validation PR-AUC
LOG_EVERY = 50      # progress line every N rounds


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def best_f1_threshold(y, p):
    """Probability threshold that maximises F1 on (y, p). Used on validation only."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, p, threshold):
    """Ranking metrics (ROC-AUC, PR-AUC) on probabilities, and classification metrics at `threshold`."""
    yhat = (p >= threshold).astype(int)
    return dict(pr_auc=average_precision_score(y, p), auc_roc=roc_auc_score(y, p),
                precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat),
                f1=f1_score(y, yhat, zero_division=0), accuracy=accuracy_score(y, yhat),
                flagged=int(yhat.sum()), threshold=float(threshold))


class _Progress:
    """Collects validation PR-AUC per round, logs it every LOG_EVERY rounds and tracks the best round."""

    def __init__(self):
        self.history, self.t0 = [], time.time()

    @property
    def best_index(self):
        return int(np.argmax(self.history))

    def add(self, value):
        self.history.append(float(value))
        n = len(self.history)
        if n == 1 or n % LOG_EVERY == 0:
            b = self.best_index
            log.info(f'    round {n:>4} | val PR-AUC {value:.5f} | best {self.history[b]:.5f} @ {b + 1} | {time.time() - self.t0:.0f}s')

    def stalled(self):
        return len(self.history) - 1 - self.best_index >= PATIENCE


# ---------------------------------------------------------------------------------------------------------------
# Encoding for XGBoost / LightGBM
# ---------------------------------------------------------------------------------------------------------------
class MatrixEncoder:
    """Turns the categorical columns into numbers. Categoricals with at most `max_onehot`
    levels (category, gender, state) are one-hot encoded; the others (merchant, job, city) are replaced by
    their frequency in the training rows. Levels and frequencies are learned in fit()."""

    def __init__(self, numeric, categorical, max_onehot=60):
        self.numeric, self.categorical, self.max_onehot = list(numeric), list(categorical), max_onehot

    def fit(self, rows):
        self.onehot_ = [c for c in self.categorical if rows[c].nunique() <= self.max_onehot]
        self.levels_ = {c: sorted(rows[c].unique()) for c in self.onehot_}
        self.freq_ = {c: rows[c].value_counts(normalize=True) for c in self.categorical if c not in self.onehot_}
        return self

    def transform(self, rows):
        X = rows[self.numeric].copy()
        for c, freq in self.freq_.items():
            X[c + '_freq'] = rows[c].map(freq).astype(float).fillna(0)      # unseen value -> 0
        for c in self.onehot_:
            dummies = pd.get_dummies(pd.Categorical(rows[c], categories=self.levels_[c]), prefix=c, dtype=float)
            X = pd.concat([X, dummies.set_index(X.index)], axis=1)
        X.columns = [str(c).replace('[', '(').replace(']', ')').replace('<', 'lt') for c in X.columns]
        return X


# ---------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------
class FraudModel:
    """Common interface and persistence. Subclasses implement fit / predict_proba / feature_importance and
    the library-specific _save_model / _load_model."""
    name = ''
    model_file = ''

    def __init__(self, numeric, categorical, seed=42):
        self.numeric, self.categorical, self.seed = list(numeric), list(categorical), seed
        self.threshold = None        # F1-maximising threshold on validation, set by train.py
        self.best_iteration = None   # number of boosting rounds kept
        self.encoder = None          # MatrixEncoder for XGBoost / LightGBM

    def fit(self, fit_df, val_df):
        raise NotImplementedError

    def predict_proba(self, df):
        raise NotImplementedError

    def feature_importance(self):
        raise NotImplementedError

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_model(directory / self.model_file)
        if self.encoder is not None:
            with open(directory / 'encoder.pkl', 'wb') as f:
                pickle.dump(self.encoder, f)
        info = dict(name=self.name, numeric=self.numeric, categorical=self.categorical, seed=self.seed,
                    threshold=self.threshold, best_iteration=self.best_iteration)
        (directory / 'model_info.json').write_text(json.dumps(info, indent=2))


class CatBoostModel(FraudModel):
    """CatBoost with native categorical features. Settings: learning rate 0.1, depth 6."""
    name, model_file = 'catboost', 'model.cbm'

    class _ValPRAUC:
        """Validation PR-AUC as CatBoost's eval metric. CatBoost may copy this object, so progress is kept on
        the class; CatBoost also evaluates the training rows, which are skipped (identified by their count)."""
        progress = None

        def __init__(self, n_val):
            self.n_val = n_val

        def is_max_optimal(self):
            return True

        def get_final_error(self, error, weight):
            return error / (weight + 1e-38)

        def evaluate(self, approxes, target, weight):
            if len(target) != self.n_val:
                return 0.0, 1.0
            ap = average_precision_score(target, 1 / (1 + np.exp(-np.asarray(approxes[0], float))))
            CatBoostModel._ValPRAUC.progress.add(ap)
            return ap, 1.0

    def _pool(self, df, with_label=True):
        from catboost import Pool
        label = df[LABEL].astype(int) if with_label else None
        return Pool(df[self.numeric + self.categorical], label, cat_features=self.categorical)

    def fit(self, fit_df, val_df):
        from catboost import CatBoostClassifier
        CatBoostModel._ValPRAUC.progress = _Progress()
        self.model = CatBoostClassifier(iterations=MAX_ITERS, learning_rate=0.1, depth=6, random_seed=self.seed,
                                        od_type='Iter', od_wait=PATIENCE, metric_period=1, thread_count=-1,
                                        eval_metric=CatBoostModel._ValPRAUC(len(val_df)),
                                        verbose=0, allow_writing_files=False)
        self.model.fit(self._pool(fit_df), eval_set=self._pool(val_df), use_best_model=True)
        self.best_iteration = self.model.get_best_iteration() + 1
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(self._pool(df, with_label=False))[:, 1]

    def feature_importance(self):
        return pd.Series(self.model.get_feature_importance(), index=self.numeric + self.categorical)

    def _save_model(self, path):
        self.model.save_model(str(path))

    def _load_model(self, path):
        from catboost import CatBoostClassifier
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))


class XGBoostModel(FraudModel):
    """XGBoost on the encoded matrix. Settings: learning rate 0.1, depth 7, 80% row/column sampling."""
    name, model_file = 'xgboost', 'model.json'

    def fit(self, fit_df, val_df):
        import xgboost as xgb
        from xgboost.callback import TrainingCallback
        self.encoder = MatrixEncoder(self.numeric, self.categorical).fit(fit_df)
        dfit = xgb.DMatrix(self.encoder.transform(fit_df), label=fit_df[LABEL].astype(int))
        dval = xgb.DMatrix(self.encoder.transform(val_df))
        y_val, progress = val_df[LABEL].to_numpy(int), _Progress()

        class EarlyStop(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                progress.add(average_precision_score(y_val, model.predict(dval)))
                return progress.stalled()            # True stops training

        params = dict(objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=7,
                      subsample=0.8, colsample_bytree=0.8, seed=self.seed)
        booster = xgb.train(params, dfit, num_boost_round=MAX_ITERS, verbose_eval=False, callbacks=[EarlyStop()])
        self.best_iteration = progress.best_index + 1
        self.model = booster[:self.best_iteration]  # keep only the rounds up to the best validation PR-AUC
        return self

    def predict_proba(self, df):
        import xgboost as xgb
        # XGBoost returns float32; float64 keeps CSV round-trips exact and matches the other models
        return self.model.predict(xgb.DMatrix(self.encoder.transform(df))).astype(np.float64)

    def feature_importance(self):
        gain = self.model.get_score(importance_type='gain')
        return pd.Series({c: gain.get(c, 0.0) for c in self.model.feature_names})

    def _save_model(self, path):
        self.model.save_model(str(path))

    def _load_model(self, path):
        import xgboost as xgb
        self.model = xgb.Booster()
        self.model.load_model(str(path))


class LightGBMModel(FraudModel):
    """LightGBM on the same encoded matrix as XGBoost. Settings mirror XGBoost: learning rate
    0.1, depth 7, 80% row/column sampling. min_sum_hessian_in_leaf and lambda_l2 match XGBoost's defaults
    (min_child_weight=1, lambda=1). With LightGBM's own defaults (1e-3, 0), leaves holding a few fraud rows get
    extreme values on this imbalanced data (in development runs, early stopping ended training after 24 rounds at
    validation PR-AUC 0.92)."""
    name, model_file = 'lightgbm', 'model.txt'

    def fit(self, fit_df, val_df):
        import lightgbm as lgb
        self.encoder = MatrixEncoder(self.numeric, self.categorical).fit(fit_df)
        dfit = lgb.Dataset(self.encoder.transform(fit_df), fit_df[LABEL].astype(int))
        dval = lgb.Dataset(self.encoder.transform(val_df), val_df[LABEL].astype(int), reference=dfit)
        progress = _Progress()
        params = dict(objective='binary', learning_rate=0.1, max_depth=7, num_leaves=127, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, min_sum_hessian_in_leaf=1.0, lambda_l2=1.0,
                      seed=self.seed, metric='None', verbose=-1)
        booster = lgb.train(params, dfit, num_boost_round=MAX_ITERS, valid_sets=[dval], valid_names=['val'],
                            feval=lambda p, d: ('pr_auc', average_precision_score(d.get_label(), p), True),
                            callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                       lambda env: progress.add(env.evaluation_result_list[0][2])])
        self.best_iteration = booster.best_iteration
        self.model = booster
        return self

    def predict_proba(self, df):
        return self.model.predict(self.encoder.transform(df), num_iteration=self.best_iteration)

    def feature_importance(self):
        return pd.Series(self.model.feature_importance('gain'), index=self.model.feature_name())

    def _save_model(self, path):
        self.model.save_model(str(path), num_iteration=self.best_iteration)

    def _load_model(self, path):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=str(path))


MODELS = {m.name: m for m in (CatBoostModel, XGBoostModel, LightGBMModel)}


def load_model(directory):
    """Load a model saved with FraudModel.save()."""
    directory = Path(directory)
    info = json.loads((directory / 'model_info.json').read_text())
    model = MODELS[info['name']](info['numeric'], info['categorical'], info['seed'])
    model.threshold, model.best_iteration = info['threshold'], info['best_iteration']
    if (directory / 'encoder.pkl').exists():
        with open(directory / 'encoder.pkl', 'rb') as f:
            model.encoder = pickle.load(f)
    model._load_model(directory / model.model_file)
    return model
