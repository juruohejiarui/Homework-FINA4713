import preprocess
import metric
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import tqdm
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

cfg : preprocess.Config = preprocess.Config.load_json("./baseline.json")

df = preprocess.load_data(cfg)

CORR_LIMIT : float = 0.9
IC_MIN_THRESHOLD : int = 30
VAL_START : pd.Timestamp = pd.Timestamp('2019-12-31')

def select_by_featurewiz(X_tr : pd.DataFrame, y_tr : pd.Series, corr_limit : float=CORR_LIMIT, n_jobs : int | None = None) -> list[str] :
    from featurewiz import FeatureWiz
    wiz = FeatureWiz(corr_limit=corr_limit, n_jobs=n_jobs)
    wiz.fit_transform(X_tr, y_tr)
    return wiz.features

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def newey_west_t(values: np.ndarray, lag: int | None = None) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return 0.0
    mean = float(values.mean())
    x = values - mean
    if lag is None:
        lag = min(int(4 * (n / 100) ** (2 / 9)), n - 1)
    lag = max(0, min(lag, n - 1))
    gamma0 = float(np.dot(x, x) / n)
    nw_var = gamma0
    for l in range(1, lag + 1):
        gamma_l = float(np.dot(x[l:], x[:-l]) / n)
        weight = 1.0 - l / (lag + 1)
        nw_var += 2.0 * weight * gamma_l
    se = np.sqrt(nw_var / n) if nw_var > 0 else 0.0
    return float(mean / se) if se > 0 else 0.0


def calc_ic(
        tr_df : pd.DataFrame,
        features : list[str],
        min_threshold : int = IC_MIN_THRESHOLD) -> pd.Series :
    out = pd.Series(index=features, dtype=float)
    for f in features :
        ic_lst : list[float] = []
        for _, g in tr_df.groupby(cfg.date_col, sort=True) :
            if len(g) < min_threshold :
                continue
            if g[f].nunique(dropna=True) <= 1 :
                continue
            ic = g[f].corr(g[cfg.target_col], method='spearman')
            if pd.isna(ic) :
                continue
            ic_lst.append(float(ic))
        out[f] = float(np.nanmean(ic_lst)) if len(ic_lst) > 0 else float('nan')
    return out


def choose_best_alpha(alpha_scores: dict[float, list[float]]) -> float:
    avg_scores = {}
    for alpha, scores in alpha_scores.items():
        if len(scores) == 0:
            avg_scores[alpha] = float('-inf')
        else:
            avg_scores[alpha] = float(np.nanmean(scores))
    best_alpha = max(avg_scores, key=lambda a: (avg_scores[a], -a))
    return best_alpha


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def pca_project(
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
        n_components: int,
        variance_ratio: float | None = None,
        max_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit PCA on training rows only; return latent matrices and effective dimension.

    If ``variance_ratio`` is set (e.g. 0.95), sklearn chooses the smallest k that
    reaches that cumulative explained variance. Otherwise use fixed ``n_components``.
    ``max_components`` caps k (after variance selection) to limit dimension blow-up.
    """
    Xt = np.asarray(X_tr, dtype=float)
    Xe = np.asarray(X_te, dtype=float)
    n_samp, n_feat = Xt.shape
    max_comp = min(n_feat, max(0, n_samp - 1))
    if max_comp < 1:
        z_tr = np.zeros((n_samp, 1))
        z_te = np.zeros((Xe.shape[0], 1))
        return z_tr, z_te, 0

    cap = max_comp
    if max_components is not None:
        cap = min(cap, max(1, max_components))

    if variance_ratio is not None:
        vr = float(variance_ratio)
        if not (0.0 < vr <= 1.0):
            raise ValueError("pca_variance must be in (0, 1]")
        pca = PCA(n_components=vr)
        z_tr = pca.fit_transform(Xt)
        z_te = pca.transform(Xe)
        k = int(z_tr.shape[1])
        if k > cap:
            z_tr = z_tr[:, :cap]
            z_te = z_te[:, :cap]
            k = cap
        return z_tr, z_te, k

    k = min(max(1, n_components), cap)
    pca = PCA(n_components=k)
    z_tr = pca.fit_transform(Xt)
    z_te = pca.transform(Xe)
    return z_tr, z_te, k


def build_selection_report(
        freq : pd.Series,
        ic_hist : dict[str, list[float]],
        feat_min_w : float = 0.01,
        feat_w_alpha : float = 0.5,
        ic_decision_eps : float = 1e-12) -> pd.DataFrame :
    report = pd.DataFrame(
        index=freq.index,
        columns=[
            'sel_freq',
            't_stat',
            'sign',
            'sign_stability',
            'dominant_sign_label',
            'ic_mean',
            'ic_std',
            'n_pos',
            'n_neg',
            'n_zero',
            'feat_w',
        ],
    )
    report['sel_freq'] = freq
    for feat, hist in ic_hist.items() :
        hist_fix = [x for x in hist if x == x]
        if len(hist_fix) == 0 :
            t_stat = 0.0
            ic_mean = float('nan')
            ic_std = float('nan')
            n_pos = 0
            n_neg = 0
            n_zero = 0
            sign_stab = float('nan')
            dominant_label = 'tie'
        else :
            t_stat = newey_west_t(np.array(hist_fix))
            ic_mean = np.mean(hist_fix)
            ic_std = np.std(hist_fix, ddof=1) if len(hist_fix) > 1 else 0.0
            n_pos = sum(1 for v in hist_fix if v > ic_decision_eps)
            n_neg = sum(1 for v in hist_fix if v < -ic_decision_eps)
            n_zero = sum(1 for v in hist_fix if abs(v) <= ic_decision_eps)
            denom = n_pos + n_neg
            sign_stab = float(max(n_pos, n_neg) / denom) if denom > 0 else float('nan')
            if n_pos > n_neg:
                dominant_label = '+'
            elif n_neg > n_pos:
                dominant_label = '-'
            else:
                dominant_label = 'tie'

        report.loc[feat, 't_stat'] = t_stat
        report.loc[feat, 'sign'] = 1 if n_pos > n_neg else (-1 if n_neg > n_pos else 0)
        report.loc[feat, 'sign_stability'] = sign_stab
        report.loc[feat, 'dominant_sign_label'] = dominant_label
        report.loc[feat, 'ic_mean'] = ic_mean
        report.loc[feat, 'ic_std'] = ic_std
        report.loc[feat, 'n_pos'] = n_pos
        report.loc[feat, 'n_neg'] = n_neg
        report.loc[feat, 'n_zero'] = n_zero

        t_score = 2 * (sigmoid(abs(t_stat)) - 0.5)
        raw_weight = feat_w_alpha * report.loc[feat, 'sel_freq'] + (1.0 - feat_w_alpha) * t_score
        report.loc[feat, 'feat_w'] = max(feat_min_w, float(raw_weight))

    return report


def apply_feature_weights(X : pd.DataFrame, feat_w : pd.Series, min_weight : float = 0.01) -> pd.DataFrame :
    weights = feat_w.reindex(X.columns).fillna(0.0).astype(float).values
    weights = np.sqrt(np.maximum(weights, 1e-12))
    return X * weights