"""Feature engineering.

There are two kinds of features:

1. History features: `add_history_features(df)`. They are computed from the transaction table alone, and
   every row only uses transactions strictly BEFORE it (by timestamp) and never a fraud label. So a feature
   can be computed at the moment a card is swiped, and the same code serves training and inference: give
   it the history plus the new transactions and it returns identical values for the new rows.

2. Job "buying power": `JobBuyingPower`. A small text model that estimates each cardholder's typical
   spending from their job title. It is fitted once on the training period (train.py), saved, and
   re-applied to new transactions (predict.py).

The merchant-similarity and merchant-location features need population statistics (which merchants share
customers, how far a merchant's customers usually are). They are running statistics over earlier transactions
with no label filter. Lookup tables over all non-fraud training rows would be simpler, but each training row's
value would then depend on its own label and on later transactions (README section 5.4; leaky_reference.py
keeps that version as a test control).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from data import CATEGORICAL, LABEL

# ---------------------------------------------------------------------------------------------------------------
# Feature groups. Every numeric model feature is listed here; NUMERIC_FEATURES fixes the column order.
# ---------------------------------------------------------------------------------------------------------------
BASE_NUM = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long',     # raw transaction / cardholder values
            'hour', 'dow', 'month', 'age', 'dist_um_km']                       # time of day/week, age, card->merchant km
USER_F = ([f'u_amt_{s}_{w}' for w in (3, 7, 30, 'all') for s in ('cnt', 'mean')]  # card's earlier spending
          + ['acct_age_days', 'amt_vs_u30', 'amt_vs_uall'])
UM_F = [f'um_cnt_{w}' for w in (3, 7, 30, 'all')]                              # earlier visits to this merchant
MP_F = ['m_amt_cnt_30', 'm_amt_mean_30', 'm_amt_cnt_all', 'm_amt_mean_all',  # merchant's earlier transactions
        'm_amt_std_all', 'amt_vs_m_all', 'amt_z_m', 'cat_amt_mean_all', 'amt_vs_cat']
JOB_F = ['job_bp_model', 'job_bp_stat', 'amt_vs_job_bp']                      # job buying power (JobBuyingPower)
SIM_F = ['sim_mean_7', 'sim_max_7', 'sim_mean_30', 'sim_max_30']             # merchant similarity
MLOC_F = ['mdist_z', 'cdist_z', 'mloc_dev_km', 'ms_pair_n']                  # merchant-location oddity
ULOC_F = [f'uloc_dist_{w}' for w in (3, 7, 30)]                               # distance from usual location
TRAVEL_F = [f'{p}_{w}' for p in ('trav_km_mean', 'trav_long_frac', 'trav_cnt') for w in (7, 30, 90)]  # travel
TSL_F = ['since_last_s', 'log_since_last', 'speed_kmh', 'since_last_merch_s', 'hop_km']               # recency

NUMERIC_FEATURES = BASE_NUM + USER_F + UM_F + MP_F + JOB_F + SIM_F + MLOC_F + ULOC_F + TRAVEL_F + TSL_F   # 63
HISTORY_FEATURES = [c for c in NUMERIC_FEATURES if c not in JOB_F]                                       # 60
CATEGORICAL_FEATURES = CATEGORICAL                                                                       # 6

# Columns kept in the feature table besides the features: identifiers, label, and inputs of JobBuyingPower
ID_COLUMNS = ['trans_num', 'cc_num', 'ts', 'trans_date_trans_time', 'source', LABEL]


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def win_stats(df, keys, val, windows, prefix, empty=-1.0):
    """For every row: count and mean of `val` over EARLIER rows with the same `keys`, within the past w days
    for each w in `windows`, plus over all earlier rows ('all'). Rows without history get `empty`.

    Implementation: sort by (group, time), take prefix sums, and find each window's start with a binary
    search on a combined (group, time) key, so it is O(n log n) for any number of groups.
    Output columns: f'{prefix}_cnt_{w}' and f'{prefix}_mean_{w}'.
    """
    gid = df.groupby(keys, sort=False).ngroup().to_numpy().astype(np.int64)
    ts = df['ts'].to_numpy()
    order = np.lexsort((ts, gid))
    g, t = gid[order], ts[order]
    comp = g * (1 << 32) + t                       # sorted composite key: (group, time)
    v = df[val].to_numpy(float)[order]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    pos = np.arange(len(df))
    start = np.searchsorted(comp, g * (1 << 32), 'left')
    out = {}
    for w in list(windows) + ['all']:
        lo = start if w == 'all' else np.maximum(np.searchsorted(comp, comp - int(w * 86400), 'left'), start)
        cnt = pos - lo                             # rows [lo, pos) = earlier rows of the group inside the window
        mean = np.where(cnt > 0, (cs[pos] - cs[lo]) / np.maximum(cnt, 1), empty)
        for name, arr in ((f'{prefix}_cnt_{w}', cnt.astype(float)), (f'{prefix}_mean_{w}', mean)):
            res = np.empty(len(df)); res[order] = arr; out[name] = res
    return pd.DataFrame(out, index=df.index)


def _add(df, new):
    for c in new.columns:
        df[c] = new[c].values


# ---------------------------------------------------------------------------------------------------------------
# History features
# ---------------------------------------------------------------------------------------------------------------
def _base(df):
    """Base: time of day/week, month, age, log amount and cardholder-to-merchant distance."""
    df['hour'] = df.dt.dt.hour
    df['dow'] = df.dt.dt.dayofweek
    df['month'] = df.dt.dt.month
    df['age'] = (df.dt - df.dob).dt.days / 365.25
    df['log_amt'] = np.log1p(df.amt)
    df['dist_um_km'] = haversine(df.lat, df.long, df.merch_lat, df.merch_long)


def _user_history(df):
    """User history: the card's earlier spending (3/7/30 days and all time), account age, amount vs card average."""
    u = win_stats(df, ['cc_num'], 'amt', [3, 7, 30], 'u_amt')
    _add(df, u)
    first_seen = df.groupby('cc_num').ts.transform('min')          # earliest transaction of the card (in the past)
    df['acct_age_days'] = np.where(u['u_amt_cnt_all'] > 0, (df.ts - first_seen) / 86400, -1)
    df['amt_vs_u30'] = np.where(df.u_amt_mean_30 > 0, df.amt / df.u_amt_mean_30.clip(lower=1e-6), -1)
    df['amt_vs_uall'] = np.where(df.u_amt_mean_all > 0, df.amt / df.u_amt_mean_all.clip(lower=1e-6), -1)


def _user_merchant_history(df):
    """User x merchant: how often this card used this merchant before (3/7/30 days and all time)."""
    df['one'] = 1.0
    _add(df, win_stats(df, ['cc_num', 'merchant'], 'one', [3, 7, 30], 'um'))


def _merchant_profiles(df):
    """Merchant profiles: the merchant's and category's earlier transactions (all cards) and this amount compared to them."""
    df['amt2'] = df.amt ** 2
    _add(df, win_stats(df, ['merchant'], 'amt', [30], 'm_amt'))
    m2 = win_stats(df, ['merchant'], 'amt2', [], 'm_amt2', empty=0.0)
    df['m_amt_std_all'] = np.sqrt(np.clip(m2['m_amt2_mean_all'] - df.m_amt_mean_all.clip(lower=0) ** 2, 0, None))
    df['amt_vs_m_all'] = np.where(df.m_amt_mean_all > 0, df.amt / df.m_amt_mean_all.clip(lower=1e-6), -1)
    df['amt_z_m'] = np.where(df.m_amt_cnt_all > 5, (df.amt - df.m_amt_mean_all) / (df.m_amt_std_all + 1), 0)
    _add(df, win_stats(df, ['category'], 'amt', [], 'cat_amt'))
    df['amt_vs_cat'] = np.where(df.cat_amt_mean_all > 0, df.amt / df.cat_amt_mean_all.clip(lower=1e-6), -1)


def sim_lookback(df, sim_of_pair):
    """For each row: mean and max similarity between its merchant and the merchants of the same card's
    transactions in the past 7 / 30 days (-1 if there are none). `sim_of_pair(k)` returns the similarity for
    every pair (row i, row i-k) of the (card, time)-sorted table."""
    cc, ts = df.cc_num.to_numpy(), df.ts.to_numpy()
    n = len(df); W7, W30 = 7 * 86400, 30 * 86400
    s7 = np.zeros(n); c7 = np.zeros(n); mx7 = np.zeros(n); s30 = np.zeros(n); c30 = np.zeros(n); mx30 = np.zeros(n)
    for k in range(1, 400):                        # look back k transactions at a time
        if k >= n:
            break
        same = cc[k:] == cc[:-k]; d = ts[k:] - ts[:-k]
        ok30 = same & (d <= W30)
        if not ok30.any():
            break
        ok7 = same & (d <= W7)
        sim = sim_of_pair(k)
        s30[k:] += np.where(ok30, sim, 0); c30[k:] += ok30; mx30[k:] = np.maximum(mx30[k:], np.where(ok30, sim, 0))
        s7[k:] += np.where(ok7, sim, 0); c7[k:] += ok7; mx7[k:] = np.maximum(mx7[k:], np.where(ok7, sim, 0))
    return {'sim_mean_7': np.where(c7 > 0, s7 / np.maximum(c7, 1), -1), 'sim_max_7': np.where(c7 > 0, mx7, -1),
            'sim_mean_30': np.where(c30 > 0, s30 / np.maximum(c30, 1), -1), 'sim_max_30': np.where(c30 > 0, mx30, -1)}


def cosine_merchant_similarity(X):
    """Merchant x merchant cosine similarity of the columns of a (card x merchant) matrix; diagonal set to 0
    because repeat visits to the same merchant are already covered by the user x merchant features (um_cnt_*)."""
    norm = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()) if sparse.issparse(X) else np.sqrt((X * X).sum(0))
    norm[norm == 0] = 1
    S = X.T @ X
    S = (S.toarray() if sparse.issparse(S) else S) / np.outer(norm, norm)
    np.fill_diagonal(S, 0)
    return S.astype(np.float32)


def _merchant_similarity(df):
    """Merchant similarity: "has this card recently shopped at merchants similar to this one?"

    Two merchants are similar when the same cards shop at both (cosine similarity of log visit counts).
    The matrix is rebuilt at the start of every calendar month from ALL transactions before that month, with no
    label filter, and each row uses its own month's matrix. Rows in the first month of the data have nothing to
    learn from and get NaN. (One matrix from all non-fraud training rows would leak; see leaky_reference.py.)
    """
    codes = {m: i for i, m in enumerate(sorted(df.merchant.unique()))}
    mid = df.merchant.map(codes).to_numpy()
    card = df.cc_num.map({c: i for i, c in enumerate(df.cc_num.unique())}).to_numpy()
    month = (df.dt.dt.year * 12 + df.dt.dt.month).to_numpy()
    month = month - month.min()
    visits = np.zeros((card.max() + 1, len(codes)))                       # card x merchant counts, months < m
    S_month = np.empty((month.max() + 1, len(codes), len(codes)), np.float32)
    for m in range(month.max() + 1):
        S_month[m] = cosine_merchant_similarity(np.log1p(visits))
        sel = month == m
        np.add.at(visits, (card[sel], mid[sel]), 1.0)                    # month m becomes history for month m+1
    for name, arr in sim_lookback(df, lambda k: S_month[month[k:], mid[k:], mid[:-k]]).items():
        arr[month == 0] = np.nan
        df[name] = arr


def _merchant_location(df):
    """Merchant location: is the cardholder-to-merchant distance or the merchant's location unusual, and how
    common is this (merchant, cardholder state) pair?

    Running statistics over transactions strictly before the row, with no label filter (lookup tables over all
    non-fraud training rows would leak; see leaky_reference.py). z-scores need more than 5 earlier rows (else 0),
    the same rule as amt_z_m. The merchant's usual location is the running mean of its earlier coordinates (a
    running median is impractical). A merchant without history gets distance 0.
    """
    df['dist2'] = df.dist_um_km ** 2
    d = df.dist_um_km.to_numpy()
    for key, out in (('merchant', 'mdist_z'), ('category', 'cdist_z')):
        mu = win_stats(df, [key], 'dist_um_km', [], 'd')
        sq = win_stats(df, [key], 'dist2', [], 'd2', empty=0.0)
        mean, cnt = mu['d_mean_all'].to_numpy(), mu['d_cnt_all'].to_numpy()
        std = np.sqrt(np.clip(sq['d2_mean_all'].to_numpy() - np.clip(mean, 0, None) ** 2, 0, None))
        df[out] = np.where(cnt > 5, (d - mean) / (std + 1), 0)
    la = win_stats(df, ['merchant'], 'merch_lat', [], 'la', empty=np.nan)
    lo = win_stats(df, ['merchant'], 'merch_long', [], 'lo', empty=np.nan)
    dev = haversine(df.merch_lat.to_numpy(), df.merch_long.to_numpy(), la['la_mean_all'].to_numpy(), lo['lo_mean_all'].to_numpy())
    df['mloc_dev_km'] = np.where(la['la_cnt_all'].to_numpy() > 0, dev, 0)
    df['ms_pair_n'] = np.log1p(win_stats(df, ['merchant', 'state'], 'one', [], 'msp')['msp_cnt_all'].to_numpy())


def _location_travel_recency(df):
    """User location, travel and recency: distance from the card's recent transaction locations, travel between
    consecutive transactions, and time since the card's / card-merchant pair's previous transaction."""
    # user location: distance from the centroid of the card's earlier transaction locations (3/7/30 days)
    lat_w = win_stats(df, ['cc_num'], 'merch_lat', [3, 7, 30], 'ul_lat', empty=np.nan)
    lon_w = win_stats(df, ['cc_num'], 'merch_long', [3, 7, 30], 'ul_lon', empty=np.nan)
    for w in (3, 7, 30):
        dist = haversine(df.merch_lat, df.merch_long, lat_w[f'ul_lat_mean_{w}'], lon_w[f'ul_lon_mean_{w}'])
        df[f'uloc_dist_{w}'] = np.where(lat_w[f'ul_lat_cnt_{w}'] > 0, dist, -1)
    # travel: "hops" between consecutive transactions of the card, summarised over 7/30/90 days
    g_u = df.groupby('cc_num')
    df['hop_km'] = haversine(df.merch_lat, df.merch_long, g_u.merch_lat.shift(), g_u.merch_long.shift()).fillna(0)
    df['hop_long'] = (df.hop_km > 50).astype(float)
    hop = win_stats(df, ['cc_num'], 'hop_km', [7, 30, 90], 'trav_km')
    hopl = win_stats(df, ['cc_num'], 'hop_long', [7, 30, 90], 'trav_long')
    for w in (7, 30, 90):
        df[f'trav_km_mean_{w}'] = hop[f'trav_km_mean_{w}']
        df[f'trav_long_frac_{w}'] = hopl[f'trav_long_mean_{w}']
        df[f'trav_cnt_{w}'] = hop[f'trav_km_cnt_{w}']
    # recency: seconds since the previous transaction of the card (-1 for its first), implied speed, card-merchant recency
    df['since_last_s'] = (df.ts - g_u.ts.shift()).fillna(-1)
    with np.errstate(divide='ignore'):          # log1p(-1) on first transactions is discarded by np.where
        df['log_since_last'] = np.where(df.since_last_s >= 0, np.log1p(df.since_last_s), -1)
    df['speed_kmh'] = np.where(df.since_last_s > 0, df.hop_km / (df.since_last_s / 3600).clip(lower=1 / 60), 0).clip(0, 2000)
    df['since_last_merch_s'] = df.groupby(['cc_num', 'merchant']).ts.diff().fillna(-1)


HISTORY_STEPS = [_base, _user_history, _user_merchant_history, _merchant_profiles,
                 _merchant_similarity, _merchant_location, _location_travel_recency]


def add_history_features(df, log=None):
    """Compute all history features for a prepared transaction table (see data.prepare).

    `df` must contain the history needed for the rows you care about. Every row only uses earlier rows, so
    adding more history never changes the past and future rows never leak into earlier ones.
    Intermediate columns are added to `df` in place (no copy, to save memory). Returns a new table with the
    identifier columns, label, categoricals, log_amt and the history features.
    """
    for step in HISTORY_STEPS:
        step(df)
        if log:
            log(f'  feature step done: {step.__name__.lstrip("_")}')
    keep = ID_COLUMNS + CATEGORICAL + ['log_amt'] + HISTORY_FEATURES
    return df[keep].copy()


# ---------------------------------------------------------------------------------------------------------------
# Job buying power
# ---------------------------------------------------------------------------------------------------------------
class JobBuyingPower:
    """Estimates a cardholder's typical spending (mean log amount) from their job title, state and age.

    Two estimates per card:
      job_bp_model  ridge regression on TF-IDF of "job title + state" plus age
      job_bp_stat   smoothed average spending of other cardholders with the same job title
    plus amt_vs_job_bp = log amount of the transaction minus job_bp_model.

    fit(rows) learns from the training-period rows. Cards seen there get OUT-OF-FOLD estimates (5 folds over
    cards), so a card's own spending never feeds its own feature; any other card is predicted by a model fitted
    on all training cards. The fitted object is pickled by train.py and re-applied by predict.py.
    It uses no fraud labels, but it learns from the whole training period, which is look-ahead for the training
    rows (a known limitation, see README).
    """

    def __init__(self, smoothing=3.0, n_folds=5):
        self.smoothing = smoothing
        self.n_folds = n_folds

    @staticmethod
    def _cards(rows, with_target):
        agg = dict(job=('job', 'first'), state=('state', 'first'), age=('age', 'first'))
        if with_target:
            agg['y'] = ('log_amt', 'mean')
        cards = rows.groupby('cc_num').agg(**agg).reset_index()
        cards['txt'] = cards.job + ' st_' + cards.state
        return cards

    @staticmethod
    def _fit_parts(cards):
        """Fit the TF-IDF + ridge model and the per-job spending table on the given cards."""
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X = sparse.hstack([vec.fit_transform(cards.txt), sparse.csr_matrix(cards[['age']].values / 100)]).tocsr()
        ridge = Ridge(alpha=1.0).fit(X, cards.y)
        job_table = cards.groupby('job').y.agg(['sum', 'count'])
        return vec, ridge, job_table, cards.y.mean()

    def _predict(self, cards, parts):
        """(ridge estimate, smoothed job mean) for each card, using fitted `parts`."""
        vec, ridge, jt, gm = parts
        X = sparse.hstack([vec.transform(cards.txt), sparse.csr_matrix(cards[['age']].values / 100)]).tocsr()
        k = self.smoothing
        stat = cards.job.map(lambda j: (jt.loc[j, 'sum'] + k * gm) / (jt.loc[j, 'count'] + k) if j in jt.index else gm)
        return ridge.predict(X), stat.to_numpy()

    def fit(self, rows):
        cards = self._cards(rows, with_target=True)
        oof = np.zeros((len(cards), 2))
        for a, b in GroupKFold(min(self.n_folds, len(cards))).split(cards, groups=cards.cc_num):
            oof[b, 0], oof[b, 1] = self._predict(cards.iloc[b], self._fit_parts(cards.iloc[a]))
        self.oof_ = pd.DataFrame(oof, index=cards.cc_num, columns=['job_bp_model', 'job_bp_stat'])
        self.parts_ = self._fit_parts(cards)            # used for cards that were not in the training period
        return self

    def transform(self, df):
        """Add job_bp_model, job_bp_stat and amt_vs_job_bp to `df` (in place) and return it."""
        cards = self._cards(df, with_target=False)
        known = cards.cc_num.isin(self.oof_.index).to_numpy()
        est = pd.DataFrame(index=cards.cc_num, columns=['job_bp_model', 'job_bp_stat'], dtype=float)
        est.loc[cards.cc_num[known]] = self.oof_.loc[cards.cc_num[known]].to_numpy()
        if (~known).any():
            p, s = self._predict(cards[~known], self.parts_)
            est.loc[cards.cc_num[~known], 'job_bp_model'] = p
            est.loc[cards.cc_num[~known], 'job_bp_stat'] = s
        df['job_bp_model'] = df.cc_num.map(est.job_bp_model).astype(float)
        df['job_bp_stat'] = df.cc_num.map(est.job_bp_stat).astype(float)
        df['amt_vs_job_bp'] = df.log_amt - df.job_bp_model
        return df
