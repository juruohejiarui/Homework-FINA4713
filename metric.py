import numpy as np
import pandas as pd
import preprocess

def calc_y_null(y_tr : np.ndarray | pd.DataFrame, y_te : np.ndarray | pd.DataFrame) -> np.ndarray :
	hist_avg = y_tr.mean()
	return np.full(len(y_te), hist_avg)

def oos_r2(y_true, y_pred, y_null) -> float :
	return float(1 - np.mean((y_true - y_pred) ** 2) / np.mean((y_true - y_null) ** 2))

def portfolio_weights(pred, max_w = 0.05) :
	n = len(pred)
	w = pd.Series(pred).rank() - (n + 1) / 2
	w /= w.abs().sum() + 1e-8
	w = w.clip(-max_w, max_w)
	w /= w.abs().sum() + 1e-8
	return w.values

def compute_portfolio_metrics(df : pd.DataFrame, pred_col : str, cfg : preprocess.Config) -> tuple[float, float, float] :
	monthly_returns = []
	for _, grp in df.groupby(cfg.date_col) :
		w = portfolio_weights(grp[pred_col].values)
		monthly_returns.append((w * grp[cfg.target_col].values).sum())
	pf = pd.Series(monthly_returns)
	ann_ret = pf.mean() * 12
	ann_vol = pf.std() * np.sqrt(12)
	sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
	return ann_ret, ann_vol, sharpe

# print oos, shape and ann. ret
def print_result(preds : dict[str, any], 
				 y_test : pd.Series, 
				 y_null_test : pd.Series, 
				 cfg : preprocess.Config,
				 names : list[str] | None = None) :
	print(f"{'Name':12} | {'Ann. Ret':>12} | {'Sharpe':>12} | {'r2':>12}")
	print("-" * (12 * 4 + 3 * 3))
	if names == None :
		# use all columns that start with 'pred_'
		names = [col[5:] for col in preds.columns if col.startswith('pred_')]
		
	for name in names :
		ann_ret, ann_vol, sharpe = compute_portfolio_metrics(preds, f"pred_{name}", cfg)
		r2 = oos_r2(y_test, preds[f'pred_{name}'], y_null_test)

		print(f"{name:12} | {ann_ret:>12.2%} | {sharpe:>12.4f} | {r2:>12.4f}")