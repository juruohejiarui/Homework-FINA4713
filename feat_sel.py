import preprocess
import metric
import pandas as pd
import numpy as np
from tqdm.auto import trange, tqdm
from featurewiz import FeatureWiz
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

cfg : preprocess.Config = preprocess.Config.load_json("./baseline.json")

df = preprocess.load_data(cfg)

def window_sel(
        df : pd.DataFrame,
        tr_size : int = 24, te_size : int = 12, w_step : int = 12) :
    freq = pd.Series([0] * len(df.columns), index=df.columns)
    tot = 0
    
    df['month'] = df['eom'].dt.to_period('M')
    all_months = sorted(df['month'].unique())
    min_month = all_months[0]
    max_month = all_months[-1]
    
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

        wiz = FeatureWiz(corr_limit=0.8, n_jobs=8)
        X_tr_sel, y_tr = wiz.fit_transform(X_tr, y_tr)
        X_te_sel = wiz.transform(X_te)

        sel_feat = wiz.features
        freq[sel_feat] += 1
        tot += 1

        feat4xgb = freq[freq / tot > 0.8].index.tolist()
        print(f"Select {len(feat4xgb)} features: ", *feat4xgb, sep='\n\t- ')


        xgb_sel = XGBRegressor(n_jobs=8)
        xgb_sel.fit(X_tr_sel, y_tr)

        xgb_all = XGBRegressor(n_jobs=8)
        xgb_all.fit(X_tr, y_tr)

        res['pred_sel'] = xgb_sel.predict(X_te_sel)
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
