import preprocess
import metric
import pandas as pd
import numpy as np
from tqdm.auto import trange, tqdm
from xgboost import XGBRegressor

cfg : preprocess.Config = preprocess.Config.load_json("./baseline.json")

df = preprocess.load_data(cfg)

FREQ_THRESHOLD : float = 0.7
SIGN_THRESHOLD : float = 0.85
IC_MIN_THRESHOLD : int = 30
IC_DECISION_EPS : float = 1e-18

def select_by_featurewiz(X_tr : pd.DataFrame, y_tr : pd.Series) -> list[str] :
    from featurewiz import FeatureWiz
    wiz = FeatureWiz(corr_limit=0.9)
    wiz.fit_transform(X_tr, y_tr)
    return wiz.features

def calc_ic(tr_df : pd.DataFrame, sel_feat : list[str]) -> pd.Series :
    out = pd.Series(index=sel_feat)
    for f in sel_feat :
        ic_lst : list[float] = []
        for _, g in tr_df.groupby(cfg.date_col, sort=True) :
            if len(g) < IC_MIN_THRESHOLD : continue
            ic = g[f].corr(g[cfg.target_col], method='spearman')
            ic_lst.append(float(ic))
        out[f] = float(np.mean(np.array(ic_lst))) if len(ic_lst) > 0 else float('nan')
    return out

def select_features(freq : pd.Series, ic_hist : dict[str, list[float]], verbose : int = 0) -> list[str] :
    report = pd.DataFrame(index=freq.index, columns=['sel_freq', 'sign_stab', 'sign', 'ic_mean', 'ic_std', 'n_pos', 'n_neg', 'n_zero'])
    report['sel_freq'] = freq
    for feat, hist in ic_hist.items() :
        hist_fix = [x for x in hist if x == x]
        if len(hist_fix) == 0 : continue
        n_pos = sum(1 for v in hist_fix if v > IC_DECISION_EPS)
        n_neg = sum(1 for v in hist_fix if v < -IC_DECISION_EPS)
        n_zero = sum(1 for v in hist_fix if abs(v) < IC_DECISION_EPS)

        stab = max(n_pos, n_neg) / (n_pos + n_neg) if n_pos + n_neg > 0 else float('nan')
        if n_pos > n_neg : sign = +1
        elif n_pos < n_neg : sign = -1
        else : sign = 0
        ic_mean = np.mean(hist_fix)
        ic_std = np.std(hist_fix, ddof=1) if len(hist_fix) > 0 else 0.0

        report.loc[feat, 'sign_stab'] = stab
        report.loc[feat, 'sign'] = sign
        report.loc[feat, 'ic_mean'] = ic_mean
        report.loc[feat, 'ic_std'] = ic_std
        report.loc[feat, 'n_pos'] = n_pos
        report.loc[feat, 'n_neg'] = n_neg
        report.loc[feat, 'n_zero'] = n_zero

    if verbose > 0 :
        print(report)
    
    return report[(report['sel_freq'] > FREQ_THRESHOLD) & (report['sign_stab'] > SIGN_THRESHOLD)].index.tolist()

def window_sel(
        df : pd.DataFrame,
        tr_size : int = 24, te_size : int = 12, w_step : int = 12) :
    tot = 0
    
    df['month'] = df['eom'].dt.to_period('M')
    all_months = sorted(df['month'].unique())
    min_month = all_months[0]
    max_month = all_months[-1]

    freq = pd.Series([0] * len(df.columns), index=df.columns)
    ic_hist : dict[str, list[float]] = {}
    for col in df.columns :
        ic_hist[str(col)] = [] 
    
    print(f"Data Range: {min_month} ~ {max_month}")

    for start in trange(0, len(all_months) - tr_size - te_size + 1, w_step) :
        tr_end = start + tr_size
        te_end = tr_end + te_size
        tr_w = all_months[start : tr_end]
        te_w = all_months[tr_end : te_end]

        tr_df, te_df = df[df['month'].isin(tr_w)], df[df['month'].isin(te_w)]
        
        tr_df, _, te_df = preprocess.transform(
            tr_df, 
            te_df.copy(),
            te_df,
            cfg, exclude=['excntry', "month"], log=False)

        X_tr, y_tr = preprocess.get_xy(tr_df, cfg, excludes=['excntry', 'month'])
        X_te, y_te = preprocess.get_xy(te_df, cfg, excludes=['excntry', 'month'])

        res = te_df[[cfg.date_col, cfg.target_col]]

        y_null = metric.calc_y_null(y_tr, y_te)

        from featurewiz import FeatureWiz

        sel_now = select_by_featurewiz(X_tr, y_tr)
        ic = calc_ic(tr_df, sel_now)
        
        freq[sel_now] += 1
        for f in df.columns :
            ic_hist[f].append(ic[f] if f in ic else float('nan'))
        
        tot += 1

        sel_feat = select_features(freq / tot, ic_hist, verbose=1)

        xgb_sel = XGBRegressor(n_jobs=8)
        xgb_sel.fit(X_tr[sel_feat], y_tr)

        xgb_all = XGBRegressor(n_jobs=8)
        xgb_all.fit(X_tr, y_tr)

        res['pred_sel'] = xgb_sel.predict(X_te[sel_feat])
        res['pred_all'] = xgb_all.predict(X_te)

        r2_sel = metric.oos_r2(y_te, res['pred_sel'], y_null)
        r2_all = metric.oos_r2(y_te, res['pred_all'], y_null)

        ann_ret_sel, _, sharpe_sel = metric.compute_portfolio_metrics(res, 'pred_sel', cfg)
        ann_ret_all, _, sharpe_all = metric.compute_portfolio_metrics(res, 'pred_all', cfg)

        tqdm.write(f"{str(all_months[start]):>10s} ~ {str(all_months[tr_end]):>10s}:")
        tqdm.write(f"all: r2: {r2_all:12.5f} ann_ret: {ann_ret_all:12.2%} sharpe: {sharpe_all:12.4f}")
        tqdm.write(f"sel: r2: {r2_sel:12.5f} ann_ret: {ann_ret_sel:12.2%} sharpe: {sharpe_sel:12.4f}")
    
    print(freq / tot)

if __name__ == "__main__" :
    cfg.train_end = pd.Timestamp("2023-12-31")
    cfg.valid_end = pd.Timestamp("2023-12-31")

    window_sel(df)
