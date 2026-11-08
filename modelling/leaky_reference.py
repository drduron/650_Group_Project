"""Deliberately LEAKY versions of the merchant-similarity and merchant-location features (SIM_F and MLOC_F in
features.py). Never used to train or score models.

They take the shortcut described in README section 5.4: lookup tables built once from ALL non-fraud rows of the
training period, then joined back onto every row, including the training rows they were built from. A
legitimate row counts itself, a fraud row was filtered out, and rows from later dates are included, so each
training row's value depends on its own fraud label and on later transactions.

check_leakage.py uses these columns as a positive control: they are known to leak, so the check must flag them.
If it does not, the check itself is broken.
"""
import numpy as np
from scipy import sparse

from features import MLOC_F, SIM_F, cosine_merchant_similarity, haversine, sim_lookback

SUFFIX = '__leaky'
LEAKY_FEATURES = [c + SUFFIX for c in SIM_F + MLOC_F]


def add_leaky_features(df, fit_mask):
    """Add the leaky columns (suffix '__leaky') to a feature table, built from the rows where `fit_mask` is True
    and is_fraud == 0."""
    fit_nonfraud = df[fit_mask & (df.is_fraud == 0)]

    # merchant similarity: one similarity matrix from all non-fraud fit rows
    codes = {m: i for i, m in enumerate(sorted(df.merchant.unique()))}
    mid = df.merchant.map(codes).to_numpy()
    cards = {c: i for i, c in enumerate(df.cc_num.unique())}
    g = fit_nonfraud.assign(mid=fit_nonfraud.merchant.map(codes)).groupby(['cc_num', 'mid']).size().reset_index(name='n')
    X = sparse.csr_matrix((np.log1p(g.n), (g.cc_num.map(cards), g.mid)), shape=(len(cards), len(codes)))
    S = cosine_merchant_similarity(X)
    for name, arr in sim_lookback(df, lambda k: S[mid[k:], mid[:-k]]).items():
        df[name + SUFFIX] = arr

    # merchant location: merchant / category / (merchant, state) tables from all non-fraud fit rows
    ms = fit_nonfraud.groupby('merchant').agg(md_mean=('dist_um_km', 'mean'), md_std=('dist_um_km', 'std'),
                                              mlat=('merch_lat', 'median'), mlon=('merch_long', 'median'))
    cs = fit_nonfraud.groupby('category').agg(cd_mean=('dist_um_km', 'mean'), cd_std=('dist_um_km', 'std'))
    pair = fit_nonfraud.groupby(['merchant', 'state']).size().rename('ms_pair_n')
    t = df[['merchant', 'category', 'state', 'dist_um_km', 'merch_lat', 'merch_long']].join(ms, on='merchant').join(cs, on='category')
    df['mdist_z' + SUFFIX] = ((t.dist_um_km - t.md_mean) / (t.md_std.fillna(0) + 1)).values
    df['cdist_z' + SUFFIX] = ((t.dist_um_km - t.cd_mean) / (t.cd_std.fillna(0) + 1)).values
    df['mloc_dev_km' + SUFFIX] = haversine(t.merch_lat, t.merch_long, t.mlat.fillna(t.merch_lat), t.mlon.fillna(t.merch_long)).values
    df['ms_pair_n' + SUFFIX] = np.log1p(df[['merchant', 'state']].join(pair, on=['merchant', 'state']).ms_pair_n.fillna(0)).values
    return df
