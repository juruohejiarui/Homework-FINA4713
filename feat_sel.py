import preprocess
import metric
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import trange, tqdm
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

cfg : preprocess.Config = preprocess.Config.load_json("./baseline.json")

df = preprocess.load_data(cfg)

CORR_LIMIT : float = 0.9
FREQ_THRESHOLD : float = 0.7
SIGN_THRESHOLD : float = 0.85
IC_MIN_THRESHOLD : int = 30
IC_DECISION_EPS : float = 1e-18
VAL_START : pd.Timestamp = pd.Timestamp('2015-12-31')

def select_by_featurewiz(X_tr : pd.DataFrame, y_tr : pd.Series, corr_limit : float=CORR_LIMIT) -> list[str] :
    from featurewiz import FeatureWiz
    wiz = FeatureWiz(corr_limit=corr_limit)
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


def choose_best_alpha(alpha_scores: dict[float, list[float]]) -> float:
    avg_scores = {}
    for alpha, scores in alpha_scores.items():
        if len(scores) == 0:
            avg_scores[alpha] = float('-inf')
        else:
            avg_scores[alpha] = float(np.nanmean(scores))
    best_alpha = max(avg_scores, key=lambda a: (avg_scores[a], -a))
    return best_alpha


def choose_best_xgb_params(param_scores: dict[tuple, list[float]], default: tuple) -> tuple:
    avg_scores = {}
    for params, scores in param_scores.items():
        if len(scores) == 0:
            avg_scores[params] = float('-inf')
        else:
            avg_scores[params] = float(np.nanmean(scores))
    if len(avg_scores) == 0:
        return default
    return max(avg_scores, key=lambda p: (avg_scores[p], -sum(p)))


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def select_features(
        freq : pd.Series,
        ic_hist : dict[str, list[float]],
        freq_threshold : float = FREQ_THRESHOLD,
        sign_threshold : float = SIGN_THRESHOLD,
        ic_decision_eps : float = IC_DECISION_EPS,
        verbose : int = 0) -> list[str] :
    report = pd.DataFrame(index=freq.index, columns=['sel_freq', 'sign_stab', 'sign', 'ic_mean', 'ic_std', 'n_pos', 'n_neg', 'n_zero'])
    report['sel_freq'] = freq
    for feat, hist in ic_hist.items() :
        hist_fix = [x for x in hist if x == x]
        if len(hist_fix) == 0 : continue
        n_pos = sum(1 for v in hist_fix if v > ic_decision_eps)
        n_neg = sum(1 for v in hist_fix if v < -ic_decision_eps)
        n_zero = sum(1 for v in hist_fix if abs(v) < ic_decision_eps)

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
    
    return report[(report['sel_freq'] > freq_threshold) & (report['sign_stab'] > sign_threshold)].index.tolist()

def window_sel(
        df : pd.DataFrame,
        tr_size : int = 24,
        te_size : int = 12,
        w_step : int = 12,
        val_start : pd.Timestamp = VAL_START,
        corr_limit : float = CORR_LIMIT,
        freq_threshold : float = FREQ_THRESHOLD,
        sign_threshold : float = SIGN_THRESHOLD,
        ic_decision_eps : float = IC_DECISION_EPS,
        ridge_alphas : list[float] | None = None,
        xgb_learning_rate : float = 0.05,
        xgb_n_estimators : int = 1000,
        xgb_stage1_max_depths : list[int] | None = None,
        xgb_stage1_min_child_weights : list[int] | None = None,
        xgb_stage2_reg_lambdas : list[float] | None = None,
        xgb_stage2_reg_alphas : list[float] | None = None) :
    tot = 0
    
    df['month'] = df['eom'].dt.to_period('M')
    all_months = sorted(df['month'].unique())
    min_month = all_months[0]
    max_month = all_months[-1]

    freq = pd.Series([0] * len(df.columns), index=df.columns)
    ic_hist : dict[str, list[float]] = {}
    for col in df.columns :
        ic_hist[str(col)] = [] 
    
    ridge_alphas = ridge_alphas if ridge_alphas is not None else [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    ridge_all_scores = {alpha: [] for alpha in ridge_alphas}
    ridge_sel_scores = {alpha: [] for alpha in ridge_alphas}
    best_ridge_all_alpha = None
    best_ridge_sel_alpha = None

    xgb_stage1_max_depths = xgb_stage1_max_depths if xgb_stage1_max_depths is not None else [2, 3, 4]
    xgb_stage1_min_child_weights = xgb_stage1_min_child_weights if xgb_stage1_min_child_weights is not None else [1, 5, 10, 20, 50]
    xgb_stage2_reg_lambdas = xgb_stage2_reg_lambdas if xgb_stage2_reg_lambdas is not None else [1, 3, 10, 30, 100]
    xgb_stage2_reg_alphas = xgb_stage2_reg_alphas if xgb_stage2_reg_alphas is not None else [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    xgb_stage1_grid = [
        (max_depth, min_child_weight)
        for max_depth in xgb_stage1_max_depths
        for min_child_weight in xgb_stage1_min_child_weights
    ]
    xgb_stage2_grid = [
        (reg_lambda, reg_alpha)
        for reg_lambda in xgb_stage2_reg_lambdas
        for reg_alpha in xgb_stage2_reg_alphas
    ]
    xgb_stage1_scores = {params: [] for params in xgb_stage1_grid}
    xgb_stage2_scores = {params: [] for params in xgb_stage2_grid}
    best_xgb_stage1_params = None
    best_xgb_stage2_params = None

    print(f"Data Range: {min_month} ~ {max_month}")

    pf : list[dict[str, float | pd.Timestamp]] = []

    start_positions = list(range(0, len(all_months) - tr_size - te_size + 1, w_step))
    final_start = len(all_months) - tr_size - te_size
    if final_start not in start_positions:
        start_positions.append(final_start)

    pre_val_positions = [pos for pos in start_positions if all_months[pos + tr_size] < val_start.to_period('M')]
    pivot = len(pre_val_positions) // 2
    xgb_stage1_positions = set(pre_val_positions[:pivot])
    xgb_stage2_positions = set(pre_val_positions[pivot:])

    for start in tqdm(start_positions) :
        tr_end = start + tr_size
        te_end = tr_end + te_size
        tr_w = all_months[start : tr_end]
        te_w = all_months[tr_end : min(len(all_months), te_end)]

        tr_df = df[df['month'].isin(tr_w)].copy()
        te_df = df[df['month'].isin(te_w)].copy()
        
        tr_df, _, te_df = preprocess.transform(
            tr_df,
            te_df.copy(),
            te_df.copy(),
            cfg, exclude=['excntry', "month"], log=False)

        X_tr, y_tr = preprocess.get_xy(tr_df, cfg, excludes=['excntry', 'month'])
        X_te, y_te = preprocess.get_xy(te_df, cfg, excludes=['excntry', 'month'])

        res = te_df[[cfg.date_col, cfg.target_col]].copy()

        y_null = metric.calc_y_null(y_tr, y_te)

        sel_now = select_by_featurewiz(X_tr, y_tr, corr_limit)
        ic = calc_ic(tr_df, sel_now)
        
        freq[sel_now] += 1
        for f in df.columns :
            ic_hist[f].append(ic[f] if f in ic else float('nan'))
        
        tot += 1

        sel_feat = select_features(
            freq / tot,
            ic_hist,
            freq_threshold=freq_threshold,
            sign_threshold=sign_threshold,
            ic_decision_eps=ic_decision_eps,
            verbose=0)

        if all_months[tr_end] < val_start.to_period('M') :
            for alpha in ridge_alphas:
                ridge_all = Ridge(alpha=alpha)
                ridge_all.fit(X_tr, y_tr)
                temp = te_df[[cfg.date_col, cfg.target_col]].copy()
                temp['pred_ridge_all'] = ridge_all.predict(X_te)
                _, _, sharpe = metric.compute_portfolio_metrics(temp, 'pred_ridge_all', cfg)
                ridge_all_scores[alpha].append(sharpe)

                if len(sel_feat) > 0:
                    ridge_sel = Ridge(alpha=alpha)
                    ridge_sel.fit(X_tr[sel_feat], y_tr)
                    temp['pred_ridge_sel'] = ridge_sel.predict(X_te[sel_feat])
                    _, _, sharpe_sel = metric.compute_portfolio_metrics(temp, 'pred_ridge_sel', cfg)
                    ridge_sel_scores[alpha].append(sharpe_sel)

            if start in xgb_stage1_positions:
                for max_depth, min_child_weight in tqdm(xgb_stage1_grid, desc='xgb stage 1'):
                    xgb = XGBRegressor(
                        learning_rate=xgb_learning_rate,
                        n_estimators=xgb_n_estimators,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        
                        use_label_encoder=False,
                        eval_metric='rmse',
                        verbosity=0)
                    xgb.fit(X_tr[sel_feat], y_tr)
                    temp = te_df[[cfg.date_col, cfg.target_col]].copy()
                    temp['pred_xgb'] = xgb.predict(X_te[sel_feat])
                    _, _, sharpe = metric.compute_portfolio_metrics(temp, 'pred_xgb', cfg)
                    xgb_stage1_scores[(max_depth, min_child_weight)].append(sharpe)
            elif start in xgb_stage2_positions:
                if best_xgb_stage1_params is None:
                    best_xgb_stage1_params = choose_best_xgb_params(xgb_stage1_scores, xgb_stage1_grid[0])
                max_depth, min_child_weight = best_xgb_stage1_params
                for reg_lambda, reg_alpha in tqdm(xgb_stage2_grid, desc='xgb stage 2'):
                    xgb = XGBRegressor(
                        learning_rate=xgb_learning_rate,
                        n_estimators=xgb_n_estimators,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha,
                        
                        use_label_encoder=False,
                        eval_metric='rmse',
                        verbosity=0)
                    xgb.fit(X_tr[sel_feat], y_tr)
                    temp = te_df[[cfg.date_col, cfg.target_col]].copy()
                    temp['pred_xgb'] = xgb.predict(X_te[sel_feat])
                    _, _, sharpe = metric.compute_portfolio_metrics(temp, 'pred_xgb', cfg)
                    xgb_stage2_scores[(reg_lambda, reg_alpha)].append(sharpe)
            continue

        if best_ridge_all_alpha is None:
            best_ridge_all_alpha = choose_best_alpha(ridge_all_scores)
            best_ridge_sel_alpha = choose_best_alpha(ridge_sel_scores)

            tqdm.write(f'best_alpha_all: {best_ridge_all_alpha:12.4f}, best_alpha_sel: {best_ridge_sel_alpha}')

        if best_xgb_stage1_params is None:
            best_xgb_stage1_params = choose_best_xgb_params(xgb_stage1_scores, xgb_stage1_grid[0])
        if best_xgb_stage2_params is None:
            best_xgb_stage2_params = choose_best_xgb_params(xgb_stage2_scores, (1.0, 0.0))

        max_depth, min_child_weight = best_xgb_stage1_params
        reg_lambda, reg_alpha = best_xgb_stage2_params

        xgb_sel = XGBRegressor(
            learning_rate=xgb_learning_rate,
            n_estimators=xgb_n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            
            use_label_encoder=False,
            eval_metric='rmse',
            verbosity=0)
        xgb_sel.fit(X_tr[sel_feat], y_tr)

        xgb_all = XGBRegressor(
            learning_rate=xgb_learning_rate,
            n_estimators=xgb_n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            
            use_label_encoder=False,
            eval_metric='rmse',
            verbosity=0)
        xgb_all.fit(X_tr, y_tr)

        ridge_all = Ridge(alpha=best_ridge_all_alpha)
        ridge_all.fit(X_tr, y_tr)

        res['pred_sel'] = xgb_sel.predict(X_te[sel_feat])
        res['pred_all'] = xgb_all.predict(X_te)
        res['pred_ridge_all'] = ridge_all.predict(X_te)

        if len(sel_feat) > 0:
            ridge_sel = Ridge(alpha=best_ridge_sel_alpha)
            ridge_sel.fit(X_tr[sel_feat], y_tr)
            res['pred_ridge_sel'] = ridge_sel.predict(X_te[sel_feat])
        else:
            res['pred_ridge_sel'] = np.zeros(len(X_te))

        r2_sel = metric.oos_r2(y_te, res['pred_sel'], y_null)
        r2_all = metric.oos_r2(y_te, res['pred_all'], y_null)
        r2_ridge_all = metric.oos_r2(y_te, res['pred_ridge_all'], y_null)
        r2_ridge_sel = metric.oos_r2(y_te, res['pred_ridge_sel'], y_null)

        ann_ret_sel, _, sharpe_sel = metric.compute_portfolio_metrics(res, 'pred_sel', cfg)
        ann_ret_all, _, sharpe_all = metric.compute_portfolio_metrics(res, 'pred_all', cfg)
        ann_ret_ridge_all, _, sharpe_ridge_all = metric.compute_portfolio_metrics(res, 'pred_ridge_all', cfg)
        ann_ret_ridge_sel, _, sharpe_ridge_sel = metric.compute_portfolio_metrics(res, 'pred_ridge_sel', cfg)

        # construct portfolio
        for eom, grp in res.groupby(cfg.date_col) :
            ret_xs = grp[cfg.target_col]
            pf.append({
                "date": eom,
                'xgboost_all': (metric.portfolio_weights(grp['pred_all'].values) * ret_xs).sum(),
                "xgboost_sel": (metric.portfolio_weights(grp['pred_sel'].values) * ret_xs).sum(),
                'ridge_all': (metric.portfolio_weights(grp['pred_ridge_all'].values) * ret_xs).sum(),
                'ridge_sel': (metric.portfolio_weights(grp['pred_ridge_sel'].values) * ret_xs).sum(),
                'market': ret_xs.mean()
            })

        tqdm.write(f"{str(all_months[start]):>10s} ~ {str(all_months[tr_end]):>10s}:")
        tqdm.write(f"xgb_all: r2: {r2_all:12.5f} ann_ret: {ann_ret_all:12.2%} sharpe: {sharpe_all:12.4f}")
        tqdm.write(f"xgb_sel: r2: {r2_sel:12.5f} ann_ret: {ann_ret_sel:12.2%} sharpe: {sharpe_sel:12.4f}")
        tqdm.write(f"ridge_all: r2: {r2_ridge_all:12.5f} ann_ret: {ann_ret_ridge_all:12.2%} sharpe: {sharpe_ridge_all:12.4f}")
        tqdm.write(f"ridge_sel: r2: {r2_ridge_sel:12.5f} ann_ret: {ann_ret_ridge_sel:12.2%} sharpe: {sharpe_ridge_sel:12.4f}")
    
    # report of selection
    select_features(
        freq / tot,
        ic_hist,
        freq_threshold=freq_threshold,
        sign_threshold=sign_threshold,
        ic_decision_eps=ic_decision_eps,
        verbose=1)

    pf : pd.DataFrame = pd.DataFrame(pf).set_index('date').sort_index()

    ann_ret = pf.mean() * 12
    ann_vol = pf.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_vol

    # plot Portfolio Construction
    cum = (1 + pf).cumprod() - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum.index, cum['market'] * 100,
            color='#888888', linestyle='--', lw=1.4,
            label=f"Market   (SR = {sharpe['market']:.2f})")
    ax.plot(cum.index, cum['xgboost_all'] * 100,
            label=f"XGB_all    (SR = {sharpe['xgboost_all']:.2f})")
    ax.plot(cum.index, cum['xgboost_sel'] * 100,
            label=f"XGB_sel    (SR = {sharpe['xgboost_sel']:.2f})")
    ax.plot(cum.index, cum['ridge_all'] * 100,
            label=f"Ridge_all  (SR = {sharpe['ridge_all']:.2f})")
    ax.plot(cum.index, cum['ridge_sel'] * 100,
            label=f"Ridge_sel  (SR = {sharpe['ridge_sel']:.2f})")
    ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.4)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylabel('Cumulative excess return (%)')
    ax.set_title(
        'Expanding-window XGBoost vs. Market — rolling\n'
        'Ridge: rank-based zero-cost long–short  |  Market: equal-weighted long-only'
    )
    ax.legend(framealpha=0.9)
    ax.grid(axis='y', lw=0.4, alpha=0.5)
    fig.tight_layout()
    plt.savefig("pf.png")

def parse_args() -> argparse.Namespace :
    parser = argparse.ArgumentParser(
        description='Feature selection and rolling XGBoost evaluation')
    parser.add_argument('--corr_limit', type=float, default=CORR_LIMIT,
                        help='Correlation limitation of FeatureWiz')
    parser.add_argument('--freq-threshold', type=float, default=FREQ_THRESHOLD,
                        help='Selection frequency threshold')
    parser.add_argument('--sign-threshold', type=float, default=SIGN_THRESHOLD,
                        help='Sign stability threshold')
    parser.add_argument('--ic-eps', type=float, default=IC_DECISION_EPS,
                        help='IC decision epsilon')
    parser.add_argument('--val-start', type=pd.Timestamp, default=VAL_START,
                        help='Validation start date')
    parser.add_argument('--tr-size', type=int, default=24,
                        help='Training window size in months')
    parser.add_argument('--te-size', type=int, default=12,
                        help='Test window size in months')
    parser.add_argument('--w-step', type=int, default=12,
                        help='Window step size in months')
    parser.add_argument('--ridge-alphas', type=parse_float_list,
                        default=np.logspace(-2, 10, 24).tolist(),
                        help='Comma-separated list of Ridge alpha candidates')
    parser.add_argument('--xgb-learning-rate', type=float, default=0.05,
                        help='XGBoost learning rate')
    parser.add_argument('--xgb-n-estimators', type=int, default=2000,
                        help='XGBoost number of estimators')
    parser.add_argument('--xgb-stage1-max-depths', type=parse_int_list,
                        default=[2, 3, 4],
                        help='Comma-separated list of XGBoost max_depth values for stage 1')
    parser.add_argument('--xgb-stage1-min-child-weights', type=parse_int_list,
                        default=[1, 5, 10, 20, 50],
                        help='Comma-separated list of XGBoost min_child_weight values for stage 1')
    parser.add_argument('--xgb-stage2-reg-lambdas', type=parse_float_list,
                        default=[1, 3, 10, 30, 100],
                        help='Comma-separated list of XGBoost reg_lambda values for stage 2')
    parser.add_argument('--xgb-stage2-reg-alphas', type=parse_float_list,
                        default=[0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
                        help='Comma-separated list of XGBoost reg_alpha values for stage 2')
    return parser.parse_args()

if __name__ == "__main__" :
    args = parse_args()
    cfg.train_end = pd.Timestamp("2023-12-31")
    cfg.valid_end = pd.Timestamp("2023-12-31")

    window_sel(
        df,
        tr_size=args.tr_size,
        te_size=args.te_size,
        w_step=args.w_step,
        val_start=args.val_start,
        freq_threshold=args.freq_threshold,
        sign_threshold=args.sign_threshold,
        ic_decision_eps=args.ic_eps,
        ridge_alphas=args.ridge_alphas,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_stage1_max_depths=args.xgb_stage1_max_depths,
        xgb_stage1_min_child_weights=args.xgb_stage1_min_child_weights,
        xgb_stage2_reg_lambdas=args.xgb_stage2_reg_lambdas,
        xgb_stage2_reg_alphas=args.xgb_stage2_reg_alphas)
