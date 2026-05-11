import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import tqdm
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from xgboost import XGBRegressor

import preprocess
import metric
from feat_sel import select_by_featurewiz, build_selection_report, calc_ic

# -----------------------------
# Config / Globals
# -----------------------------
cfg: preprocess.Config = preprocess.Config.load_json("./baseline.json")
df = preprocess.load_data(cfg)

VAL_START: pd.Timestamp = pd.Timestamp("2019-01-01")
IC_MIN_THRESHOLD: int = 30

# -----------------------------
# Utils
# -----------------------------
def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def choose_best_alpha(alpha_scores: dict[float, list[float]]) -> float:
    avg_scores = {}
    for a, lst in alpha_scores.items():
        avg_scores[a] = float(np.nanmean(lst)) if len(lst) else float("-inf")
    # tie-break: prefer smaller alpha if same score
    best = max(avg_scores, key=lambda x: (avg_scores[x], -x))
    return float(best)

def choose_best_pair(scores: dict[tuple, list[float]]) -> tuple:
    avg = {}
    for k, lst in scores.items():
        avg[k] = float(np.nanmean(lst)) if len(lst) else float("-inf")
    best = max(avg, key=lambda x: avg[x])
    return best

def calc_ic_metrics(df_res: pd.DataFrame, pred_col: str, target_col: str, date_col: str) -> tuple[float, float, float]:
    """计算单个模型的截面 IC (Spearman Rank Correlation) 的均值、标准差和 t-stat"""
    def _spearman(sub_df):
        if len(sub_df) < 2: 
            return np.nan
        # 如果预测值或真实值全一样，corr会报nan错误
        if sub_df[pred_col].nunique() <= 1 or sub_df[target_col].nunique() <= 1:
            return np.nan
        return sub_df[pred_col].corr(sub_df[target_col], method="spearman")

    if pred_col not in df_res.columns or target_col not in df_res.columns:
        return np.nan, np.nan, np.nan

    ics = df_res.groupby(date_col).apply(_spearman).dropna()
    if len(ics) == 0:
        return np.nan, np.nan, np.nan

    ic_mean = float(ics.mean())
    ic_std = float(ics.std())
    # t-stat = mean / standard_error = mean / (std / sqrt(N))
    t_stat = float(ic_mean / (ic_std / np.sqrt(len(ics)))) if ic_std > 0 else np.nan
    
    return ic_mean, ic_std, t_stat

# -----------------------------
# Factor zoo: clustering -> cluster factors
# -----------------------------
def _spearman_corr_for_features(X: pd.DataFrame) -> np.ndarray:
    R = X.rank(axis=0, method="average", na_option="keep")
    C = R.corr(method="pearson").values
    C = np.nan_to_num(C, nan=0.0)
    C = (C + C.T) * 0.5
    np.fill_diagonal(C, 1.0)
    return C

def cluster_features(X_tr: pd.DataFrame, n_clusters: int) -> dict[int, list[str]]:
    feats = list(X_tr.columns)
    C = _spearman_corr_for_features(X_tr)
    D = 1.0 - np.abs(C) 
    D = np.clip(D, 0.0, 2.0)

    cl = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = cl.fit_predict(D)

    groups: dict[int, list[str]] = {}
    for f, lab in zip(feats, labels):
        groups.setdefault(int(lab), []).append(f)
    return groups

class ClusterFactorizer:
    def __init__(self, groups: dict[int, list[str]], method: str):
        self.groups = groups
        self.method = method
        self.pcas: dict[int, PCA] = {}

    def fit(self, X_tr: pd.DataFrame):
        if "pc1" in self.method:
            for k, feats in self.groups.items():
                if len(feats) < 2:
                    continue
                pca = PCA(n_components=1, random_state=0)
                pca.fit(X_tr[feats].values)
                self.pcas[k] = pca
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out_cols = {}
        for k, feats in self.groups.items():
            Xg = X[feats]
            if "mean" in self.method:
                out_cols[f"cl{k:02d}_mean"] = Xg.mean(axis=1)

            if "pc1" in self.method:
                if len(feats) == 1:
                    out_cols[f"cl{k:02d}_pc1"] = Xg.iloc[:, 0]
                else:
                    pca = self.pcas.get(k, None)
                    if pca is None:
                        out_cols[f"cl{k:02d}_pc1"] = Xg.mean(axis=1)
                    else:
                        z = pca.transform(Xg.values).reshape(-1)
                        out_cols[f"cl{k:02d}_pc1"] = z

        Z = pd.DataFrame(out_cols, index=X.index)
        return Z

def build_factor_zoo(X_tr: pd.DataFrame, X_te: pd.DataFrame, n_clusters: int, method: str = "mean+pc1"):
    groups = cluster_features(X_tr, n_clusters=n_clusters)
    fac = ClusterFactorizer(groups, method=method).fit(X_tr)
    Z_tr = fac.transform(X_tr)
    Z_te = fac.transform(X_te)
    return Z_tr, Z_te, groups

# -----------------------------
# Main rolling evaluation
# -----------------------------
def window_factor_zoo(
    df: pd.DataFrame,
    tr_size: int = 24,
    te_size: int = 12,
    w_step: int = 12,
    val_start: pd.Timestamp = VAL_START,
    n_clusters: int = 10,
    pca_method: str = "mean+pc1",
    ridge_alphas: list[float] | None = None,
    xgb_max_depths: list[int] | None = None,
    xgb_min_child_weights: list[float] | None = None,
    featurewiz_corr_limit: float = 0.9,
    feat_min_w: float = 0.01,
    feat_w_alpha: float = 0.5,
    ic_min_threshold: int = IC_MIN_THRESHOLD,
    ic_decision_eps: float = 1e-12,
    xgb_learning_rate: float = 0.05,
    xgb_n_estimators: int = 800,
    output_iden: str = "",
    n_jobs: int | None = None,
):
    df = df.copy()
    df["month"] = df["eom"].dt.to_period("M")
    all_months = sorted(df["month"].unique())
    min_month, max_month = all_months[0], all_months[-1]

    suffix = f"_{output_iden}" if output_iden else ""
    info_line = f"Data Range: {min_month} ~ {max_month}"
    print(info_line)

    hp_lines: list[str] = [] # 记录保留在txt中的超参数
    window_metrics_list = [] # 记录提取成csv的窗口评估结果
    valid_res_list = []      # 记录val_start之后的res做整体IC评估

    ridge_alphas = ridge_alphas if ridge_alphas is not None else np.logspace(-3, 6, 16).tolist()
    xgb_max_depths = xgb_max_depths if xgb_max_depths is not None else [2, 3, 4, 5]
    xgb_min_child_weights = xgb_min_child_weights if xgb_min_child_weights is not None else [1.0, 3.0, 5.0, 10.0]

    ridge_scores_all = {a: [] for a in ridge_alphas}
    ridge_scores_pca = {a: [] for a in ridge_alphas}
    xgb_scores_all = {(d, mcw): [] for d in xgb_max_depths for mcw in xgb_min_child_weights}
    xgb_scores_pca = {(d, mcw): [] for d in xgb_max_depths for mcw in xgb_min_child_weights}
    xgb_scores_sel = {(d, mcw): [] for d in xgb_max_depths for mcw in xgb_min_child_weights}
    ridge_sel_scores = {a: [] for a in ridge_alphas}

    best_ridge_alpha_all = None
    best_ridge_alpha_pca = None
    best_ridge_sel_alpha = None
    best_xgb_pair_all = None
    best_xgb_pair_pca = None
    best_xgb_pair_sel = None

    feature_cols: list[str] | None = None
    freq: pd.Series | None = None
    ic_hist: dict[str, list[float]] = {}
    tot = 0

    pf_rows: list[dict[str, float | pd.Timestamp]] = []
    feature_names: list[str] | None = None
    feat_idx: dict[str, int] | None = None
    same_group_counts: np.ndarray | None = None
    n_group_windows = 0

    start_positions = list(range(0, len(all_months) - tr_size - te_size + 1, w_step))
    final_start = len(all_months) - tr_size - te_size
    if final_start not in start_positions:
        start_positions.append(final_start)

    for start in tqdm(start_positions, desc="Rolling windows"):
        tr_end = start + tr_size
        te_end = min(len(all_months), tr_end + te_size)
        tr_w = all_months[start:tr_end]
        te_w = all_months[tr_end:te_end]

        tr_df = df[df["month"].isin(tr_w)].copy()
        te_df = df[df["month"].isin(te_w)].copy()

        tr_df, _, te_df = preprocess.transform(
            tr_df, te_df.copy(), te_df.copy(), cfg, exclude=["excntry", "month"], log=False
        )

        X_tr, y_tr = preprocess.get_xy(tr_df, cfg, excludes=["excntry", "month"])
        X_te, y_te = preprocess.get_xy(te_df, cfg, excludes=["excntry", "month"])

        if feature_cols is None:
            feature_cols = list(X_tr.columns)
            freq = pd.Series(0.0, index=feature_cols)
            ic_hist = {feat: [] for feat in feature_cols}

        sel_now = select_by_featurewiz(X_tr, y_tr, corr_limit=featurewiz_corr_limit, n_jobs=n_jobs)
        ic = calc_ic(tr_df, feature_cols, min_threshold=ic_min_threshold)
        if len(sel_now) > 0:
            freq.loc[sel_now] += 1.0
        for f in feature_cols:
            ic_hist[f].append(ic.get(f, float("nan")))
        tot += 1

        n_cl = int(min(max(2, n_clusters), X_tr.shape[1]))
        Z_tr, Z_te, _groups = build_factor_zoo(X_tr, X_te, n_clusters=n_cl, method=pca_method)

        if feature_names is None:
            feature_names = list(X_tr.columns)
            p = len(feature_names)
            same_group_counts = np.zeros((p, p), dtype=float)
            feat_idx = {f: i for i, f in enumerate(feature_names)}

        if same_group_counts is not None:
            for feats in _groups.values():
                for i, fi in enumerate(feats):
                    for fj in feats[i:]:
                        idx_i = feat_idx[fi]
                        idx_j = feat_idx[fj]
                        same_group_counts[idx_i, idx_j] += 1.0
                        if idx_i != idx_j:
                            same_group_counts[idx_j, idx_i] += 1.0
            n_group_windows += 1

        res = te_df[[cfg.date_col, cfg.target_col]].copy()
        y_null = metric.calc_y_null(y_tr, y_te)

        # validation phase
        if all_months[te_end - 1] < val_start.to_period("M"):
            for a in tqdm(ridge_alphas, desc="Ridge"):
                m_all = Ridge(alpha=a)
                m_all.fit(X_tr, y_tr)
                tmp = res.copy()
                tmp["pred"] = m_all.predict(X_te)
                _, _, sr = metric.compute_portfolio_metrics(tmp, "pred", cfg)
                ridge_scores_all[a].append(sr)

                m_pca = Ridge(alpha=a)
                m_pca.fit(Z_tr, y_tr)
                tmp2 = res.copy()
                tmp2["pred"] = m_pca.predict(Z_te)
                _, _, sr2 = metric.compute_portfolio_metrics(tmp2, "pred", cfg)
                ridge_scores_pca[a].append(sr2)

            if len(sel_now) > 0:
                for a in tqdm(ridge_alphas, desc="Ridge_sel"):
                    m_sel = Ridge(alpha=a)
                    m_sel.fit(X_tr[sel_now], y_tr)
                    tmp = res.copy()
                    tmp["pred_ridge_sel"] = m_sel.predict(X_te[sel_now])
                    _, _, sr_sel = metric.compute_portfolio_metrics(tmp, "pred_ridge_sel", cfg)
                    ridge_sel_scores[a].append(sr_sel)

            for (d, mcw) in tqdm(itertools.product(xgb_max_depths, xgb_min_child_weights), desc="XGB", total=len(xgb_max_depths)*len(xgb_min_child_weights)):
                m_all = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(d), min_child_weight=float(mcw), n_jobs=n_jobs, objective="reg:squarederror")
                m_all.fit(X_tr, y_tr)
                tmp = res.copy()
                tmp["pred"] = m_all.predict(X_te)
                _, _, sr = metric.compute_portfolio_metrics(tmp, "pred", cfg)
                xgb_scores_all[(d, mcw)].append(sr)

                m_pca = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(d), min_child_weight=float(mcw), n_jobs=n_jobs, objective="reg:squarederror")
                m_pca.fit(Z_tr, y_tr)
                tmp2 = res.copy()
                tmp2["pred"] = m_pca.predict(Z_te)
                _, _, sr2 = metric.compute_portfolio_metrics(tmp2, "pred", cfg)
                xgb_scores_pca[(d, mcw)].append(sr2)

                if len(sel_now) > 0:
                    m_sel = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(d), min_child_weight=float(mcw), n_jobs=n_jobs, objective="reg:squarederror")
                    m_sel.fit(X_tr[sel_now], y_tr)
                    tmp3 = res.copy()
                    tmp3["pred_xgb_sel"] = m_sel.predict(X_te[sel_now])
                    _, _, sr3 = metric.compute_portfolio_metrics(tmp3, "pred_xgb_sel", cfg)
                    xgb_scores_sel[(d, mcw)].append(sr3)
            continue

        # after val_start
        if best_ridge_alpha_all is None:
            best_ridge_alpha_all = choose_best_alpha(ridge_scores_all)
            best_ridge_alpha_pca = choose_best_alpha(ridge_scores_pca)
            best_ridge_sel_alpha = choose_best_alpha(ridge_sel_scores)

            best_xgb_pair_all = choose_best_pair(xgb_scores_all)
            best_xgb_pair_pca = choose_best_pair(xgb_scores_pca)
            best_xgb_pair_sel = choose_best_pair(xgb_scores_sel)

            line = (
                f"Chosen HPs @ val_start={val_start.date()} | "
                f"Ridge(all)={best_ridge_alpha_all:.4g}, Ridge(pca)={best_ridge_alpha_pca:.4g}, Ridge(sel)={best_ridge_sel_alpha:.4g} | "
                f"XGB(all)=(d={best_xgb_pair_all[0]}, mcw={best_xgb_pair_all[1]}), "
                f"XGB(pca)=(d={best_xgb_pair_pca[0]}, mcw={best_xgb_pair_pca[1]}), "
                f"XGB(sel)=(d={best_xgb_pair_sel[0]}, mcw={best_xgb_pair_sel[1]})"
            )
            tqdm.write(line)
            hp_lines.append(line)

        # Fit final models
        ridge_all = Ridge(alpha=best_ridge_alpha_all).fit(X_tr, y_tr)
        ridge_pca = Ridge(alpha=best_ridge_alpha_pca).fit(Z_tr, y_tr)
        ridge_sel = Ridge(alpha=best_ridge_sel_alpha).fit(X_tr[sel_now], y_tr) if len(sel_now) > 0 else None

        xgb_all = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(best_xgb_pair_all[0]), min_child_weight=float(best_xgb_pair_all[1]), n_jobs=n_jobs).fit(X_tr, y_tr)
        xgb_pca = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(best_xgb_pair_pca[0]), min_child_weight=float(best_xgb_pair_pca[1]), n_jobs=n_jobs).fit(Z_tr, y_tr)
        xgb_sel = XGBRegressor(learning_rate=xgb_learning_rate, n_estimators=xgb_n_estimators, max_depth=int(best_xgb_pair_sel[0]), min_child_weight=float(best_xgb_pair_sel[1]), n_jobs=n_jobs, objective="reg:squarederror").fit(X_tr[sel_now], y_tr) if len(sel_now) > 0 else None

        res["pred_ridge_all"] = ridge_all.predict(X_te)
        res["pred_ridge_pca"] = ridge_pca.predict(Z_te)
        res["pred_ridge_sel"] = ridge_sel.predict(X_te[sel_now]) if ridge_sel is not None else np.zeros(len(X_te))
        res["pred_xgb_all"] = xgb_all.predict(X_te)
        res["pred_xgb_pca"] = xgb_pca.predict(Z_te)
        res["pred_xgb_sel"] = xgb_sel.predict(X_te[sel_now]) if xgb_sel is not None else np.zeros(len(X_te))

        # Metrics
        r2_ridge_all = metric.oos_r2(y_te, res["pred_ridge_all"], y_null)
        r2_ridge_pca = metric.oos_r2(y_te, res["pred_ridge_pca"], y_null)
        r2_ridge_sel = metric.oos_r2(y_te, res["pred_ridge_sel"], y_null)
        r2_xgb_all = metric.oos_r2(y_te, res["pred_xgb_all"], y_null)
        r2_xgb_pca = metric.oos_r2(y_te, res["pred_xgb_pca"], y_null)
        r2_xgb_sel = metric.oos_r2(y_te, res["pred_xgb_sel"], y_null)

        ann_ret_ra, _, sr_ra = metric.compute_portfolio_metrics(res, "pred_ridge_all", cfg)
        ann_ret_rp, _, sr_rp = metric.compute_portfolio_metrics(res, "pred_ridge_pca", cfg)
        ann_ret_rs, _, sr_rs = metric.compute_portfolio_metrics(res, "pred_ridge_sel", cfg)
        ann_ret_xa, _, sr_xa = metric.compute_portfolio_metrics(res, "pred_xgb_all", cfg)
        ann_ret_xp, _, sr_xp = metric.compute_portfolio_metrics(res, "pred_xgb_pca", cfg)
        ann_ret_xs, _, sr_xs = metric.compute_portfolio_metrics(res, "pred_xgb_sel", cfg)

        # Window IC Stats
        ic_ra_m, ic_ra_s, ts_ra = calc_ic_metrics(res, "pred_ridge_all", cfg.target_col, cfg.date_col)
        ic_rp_m, ic_rp_s, ts_rp = calc_ic_metrics(res, "pred_ridge_pca", cfg.target_col, cfg.date_col)
        ic_rs_m, ic_rs_s, ts_rs = calc_ic_metrics(res, "pred_ridge_sel", cfg.target_col, cfg.date_col)
        ic_xa_m, ic_xa_s, ts_xa = calc_ic_metrics(res, "pred_xgb_all", cfg.target_col, cfg.date_col)
        ic_xp_m, ic_xp_s, ts_xp = calc_ic_metrics(res, "pred_xgb_pca", cfg.target_col, cfg.date_col)
        ic_xs_m, ic_xs_s, ts_xs = calc_ic_metrics(res, "pred_xgb_sel", cfg.target_col, cfg.date_col)

        def add_window_metric(model_name, r2, ret, sr, ic_m, ic_s, ts):
            window_metrics_list.append({
                "start_month": str(all_months[start]),
                "tr_end_month": str(all_months[tr_end]),
                "model": model_name,
                "r2": r2, "ann_ret": ret, "sharpe": sr,
                "ic_mean": ic_m, "ic_std": ic_s, "t_stat": ts
            })

        add_window_metric("Ridge_all", r2_ridge_all, ann_ret_ra, sr_ra, ic_ra_m, ic_ra_s, ts_ra)
        add_window_metric("Ridge_pca", r2_ridge_pca, ann_ret_rp, sr_rp, ic_rp_m, ic_rp_s, ts_rp)
        add_window_metric("Ridge_sel", r2_ridge_sel, ann_ret_rs, sr_rs, ic_rs_m, ic_rs_s, ts_rs)
        add_window_metric("XGB_all",   r2_xgb_all,   ann_ret_xa, sr_xa, ic_xa_m, ic_xa_s, ts_xa)
        add_window_metric("XGB_pca",   r2_xgb_pca,   ann_ret_xp, sr_xp, ic_xp_m, ic_xp_s, ts_xp)
        add_window_metric("XGB_sel",   r2_xgb_sel,   ann_ret_xs, sr_xs, ic_xs_m, ic_xs_s, ts_xs)

        line1 = f"{str(all_months[start]):>10s} ~ {str(all_months[tr_end]):>10s}:"
        line2 = f"Ridge_all: r2 {r2_ridge_all:12.5f} ann_ret {ann_ret_ra:12.2%} sharpe {sr_ra:12.3f} IC_M {ic_ra_m:.3f}"
        line3 = f"Ridge_pca: r2 {r2_ridge_pca:12.5f} ann_ret {ann_ret_rp:12.2%} sharpe {sr_rp:12.3f} IC_M {ic_rp_m:.3f}"
        line4 = f"Ridge_sel: r2 {r2_ridge_sel:12.5f} ann_ret {ann_ret_rs:12.2%} sharpe {sr_rs:12.3f} IC_M {ic_rs_m:.3f}"
        line5 = f"XGB_all:   r2 {r2_xgb_all:12.5f} ann_ret {ann_ret_xa:12.2%} sharpe {sr_xa:12.3f} IC_M {ic_xa_m:.3f}"
        line6 = f"XGB_pca:   r2 {r2_xgb_pca:12.5f} ann_ret {ann_ret_xp:12.2%} sharpe {sr_xp:12.3f} IC_M {ic_xp_m:.3f}"
        line7 = f"XGB_sel:   r2 {r2_xgb_sel:12.5f} ann_ret {ann_ret_xs:12.2%} sharpe {sr_xs:12.3f} IC_M {ic_xs_m:.3f}"
        for ln in [line1, line2, line3, line4, line5, line6, line7]:
            tqdm.write(ln)

        for eom, grp in res.groupby(cfg.date_col):
            if eom < val_start:
                continue
            valid_res_list.append(grp) # 保存下来用于计算整体的IC
            ret_xs = grp[cfg.target_col]
            pf_rows.append({
                "date": eom,
                "ridge_all": (metric.portfolio_weights(grp["pred_ridge_all"].values) * ret_xs).sum(),
                "ridge_pca": (metric.portfolio_weights(grp["pred_ridge_pca"].values) * ret_xs).sum(),
                "ridge_sel": (metric.portfolio_weights(grp["pred_ridge_sel"].values) * ret_xs).sum(),
                "xgb_all":   (metric.portfolio_weights(grp["pred_xgb_all"].values)   * ret_xs).sum(),
                "xgb_pca":   (metric.portfolio_weights(grp["pred_xgb_pca"].values)   * ret_xs).sum(),
                "xgb_sel":   (metric.portfolio_weights(grp["pred_xgb_sel"].values)   * ret_xs).sum(),
                "market":    ret_xs.mean(),
            })

    # -------- save metrics and outputs --------
    # 1. Window metrics CSV 
    window_metrics_df = pd.DataFrame(window_metrics_list)
    window_csv_path = f"window_metrics{suffix}.csv"
    if not window_metrics_df.empty:
        window_metrics_df.to_csv(window_csv_path, index=False)

    # 2. Portfolio cumulative calculation & summary 
    pf = pd.DataFrame(pf_rows).set_index("date").sort_index()
    pf.index = pd.to_datetime(pf.index)
    if pf.index.duplicated().any():
        pf = pf.groupby(level=0).mean()

    ann_ret = pf.mean() * 12
    ann_vol = pf.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol

    # 计算整体IC（汇总所有的 test window preds）
    summary_df = pd.DataFrame({"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe})
    summary_df["ic_mean"] = np.nan
    summary_df["ic_std"] = np.nan
    summary_df["t_stat"] = np.nan

    overall_res = pd.concat(valid_res_list, ignore_index=True) if len(valid_res_list) > 0 else pd.DataFrame()
    if not overall_res.empty:
        models = ["ridge_all", "ridge_pca", "ridge_sel", "xgb_all", "xgb_pca", "xgb_sel"]
        for m in models:
            p_col = f"pred_{m}"
            if p_col in overall_res.columns:
                m_ic_m, m_ic_s, m_ts = calc_ic_metrics(overall_res, p_col, cfg.target_col, cfg.date_col)
                if m in summary_df.index:
                    summary_df.loc[m, "ic_mean"] = m_ic_m
                    summary_df.loc[m, "ic_std"] = m_ic_s
                    summary_df.loc[m, "t_stat"] = m_ts

    summary_csv_path = f"summary_metrics{suffix}.csv"
    summary_df.to_csv(summary_csv_path)

    # 3. Plot
    cum = (1.0 + pf).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(cum.index, cum["market"] * 100, linestyle="--", lw=1.4, label=f"Market (SR={sharpe['market']:4.2f})")
    ax.plot(cum.index, cum["ridge_all"] * 100, label=f"Ridge_all (SR={sharpe['ridge_all']:4.2f})")
    ax.plot(cum.index, cum["ridge_pca"] * 100, label=f"Ridge_pca (SR={sharpe['ridge_pca']:4.2f})")
    ax.plot(cum.index, cum["ridge_sel"] * 100, label=f"Ridge_sel (SR={sharpe['ridge_sel']:4.2f})")
    ax.plot(cum.index, cum["xgb_all"]   * 100, label=f"XGB_all   (SR={sharpe['xgb_all']:4.2f})")
    ax.plot(cum.index, cum["xgb_pca"]   * 100, label=f"XGB_pca   (SR={sharpe['xgb_pca']:4.2f})")
    ax.plot(cum.index, cum["xgb_sel"]   * 100, label=f"XGB_sel   (SR={sharpe['xgb_sel']:4.2f})")
    ax.axhline(0, lw=0.8, linestyle="--", alpha=0.4)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Cumulative excess return (%)")
    ax.set_title("Rolling-window Models: All features vs Factor zoo")
    ax.legend(framealpha=0.9, ncol=2)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    fig.tight_layout()

    png_path = f"pf{suffix}.png"
    plt.savefig(png_path, dpi=300)
    
    pf_csv_path = f"pf{suffix}.csv"
    pf.to_csv(pf_csv_path)

    # 4. 纯净版 TXT：包含最佳超参数与 Selection report
    txt_path = f"result{suffix}.txt"
    selection_report = build_selection_report(
        freq / tot,
        ic_hist,
        feat_min_w=feat_min_w,
        feat_w_alpha=feat_w_alpha,
        ic_decision_eps=ic_decision_eps,
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(info_line + "\n\n")
        f.write("=== Hyperparameters ===\n")
        f.write("\n".join(hp_lines) + "\n\n")
        f.write("=== Selection report ===\n")
        f.write(selection_report.to_string() + "\n")

    selection_csv_path = f"selection_report{suffix}.csv"
    selection_report.to_csv(selection_csv_path)

    print(f"Saved window metrics to {window_csv_path}")
    print(f"Saved summary metrics to {summary_csv_path}")
    print(f"Saved selection report CSV to {selection_csv_path}")
    print(f"Saved image to {png_path}")
    print(f"Saved portfolio CSV to {pf_csv_path}")
    print(f"Saved txt results to {txt_path}")

    # Heatmap code
    if same_group_counts is not None and n_group_windows > 0:
        same_group_freq = same_group_counts / float(n_group_windows)
        same_group_df = pd.DataFrame(same_group_freq, index=feature_names, columns=feature_names)

        if len(feature_names) > 1:
            dist = 1.0 - same_group_df.values
            dist = np.clip(dist, 0.0, 2.0)
            dist = (dist + dist.T) * 0.5
            condensed = squareform(dist)
            linkage_matrix = linkage(condensed, method="average")
            order = leaves_list(linkage_matrix)
            ordered_names = [feature_names[i] for i in order]
            same_group_df = same_group_df.reindex(index=ordered_names, columns=ordered_names)
        else:
            ordered_names = feature_names

        figsize_scale = max(12, len(feature_names) * 0.14)
        g = sns.clustermap(
            same_group_df,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            row_cluster=False,
            col_cluster=False,
            cbar_pos=(0.02, 0.8, 0.03, 0.18),
            xticklabels=True,
            yticklabels=True,
            figsize=(figsize_scale, figsize_scale),
        )
        g.ax_heatmap.set_xticklabels(ordered_names, rotation=90,
                                     fontsize=max(4, min(10, 240 // len(ordered_names))))
        g.ax_heatmap.set_yticklabels(ordered_names,
                                     fontsize=max(4, min(10, 240 // len(ordered_names))))
        g.ax_heatmap.set_title("Feature pair same-cluster frequency")
        g.ax_heatmap.set_xlabel("Feature")
        g.ax_heatmap.set_ylabel("Feature")
        g.figure.tight_layout()
        same_group_csv_path = f"same_group_freq{suffix}.csv"
        same_group_df.to_csv(same_group_csv_path)
        heatmap_path = f"group_cooccurrence{suffix}.png"
        g.figure.savefig(heatmap_path, dpi=300)
        plt.close(g.figure)
        print(f"Saved same-group frequency CSV to {same_group_csv_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Rolling window factor zoo prediction")
    p.add_argument("--tr-size", type=int, default=24)
    p.add_argument("--te-size", type=int, default=12)
    p.add_argument("--w-step", type=int, default=12)
    p.add_argument("--val-start", type=pd.Timestamp, default=VAL_START)

    p.add_argument("--n-clusters", type=int, default=10)
    p.add_argument("--zoo-method", "--pca-method", dest="pca_method", type=str, default="mean+pc1", 
                   choices=["mean", "pc1", "mean+pc1"],
                   help="Factor PCA method for cluster factors: mean, pc1, or mean+pc1")

    p.add_argument("--ridge-alphas", type=parse_float_list, default=np.logspace(-2, 10, 24))
    p.add_argument("--featurewiz-corr-limit", type=float, default=0.9)
    p.add_argument("--feat-min-w", type=float, default=0.01)
    p.add_argument("--feat-w-alpha", type=float, default=0.5)
    p.add_argument("--ic-min-threshold", type=int, default=IC_MIN_THRESHOLD)
    p.add_argument("--ic-decision-eps", type=float, default=1e-12)

    p.add_argument("--xgb-max-depths", type=parse_int_list, default="2,3,4")
    p.add_argument("--xgb-min-child-weights", type=parse_float_list, default="1,3,5")
    p.add_argument("--xgb-learning-rate", type=float, default=0.05)
    p.add_argument("--xgb-n-estimators", type=int, default=200)

    p.add_argument("--output-iden", type=str, default="")
    p.add_argument("--n-jobs", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg.train_end = pd.Timestamp("2023-12-31")
    cfg.valid_end = pd.Timestamp("2023-12-31")

    window_factor_zoo(
        df,
        tr_size=args.tr_size,
        te_size=args.te_size,
        w_step=args.w_step,
        val_start=args.val_start,
        n_clusters=args.n_clusters,
        pca_method=args.pca_method,
        ridge_alphas=args.ridge_alphas,
        xgb_max_depths=args.xgb_max_depths,
        xgb_min_child_weights=args.xgb_min_child_weights,
        featurewiz_corr_limit=args.featurewiz_corr_limit,
        feat_min_w=args.feat_min_w,
        feat_w_alpha=args.feat_w_alpha,
        ic_min_threshold=args.ic_min_threshold,
        ic_decision_eps=args.ic_decision_eps,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_n_estimators=args.xgb_n_estimators,
        output_iden=args.output_iden,
        n_jobs=args.n_jobs,
    )