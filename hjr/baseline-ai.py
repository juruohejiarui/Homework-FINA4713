import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline pipeline for JKP dataset')
    parser.add_argument('--data_path', type=str, default='jkp_data.parquet', help='Path to JKP parquet file')
    return parser.parse_args()


def load_data(path):
    df = pd.read_parquet(path)
    required = {'id', 'eom', 'ret_exc_lead1m'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing required columns: {missing}')
    df['eom'] = pd.to_datetime(df['eom'])
    return df


def build_feature_list(df):
    exclude = {'id', 'eom', 'ret_exc_lead1m', 'excntry'}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric if c not in exclude]
    return sorted(features)


def prepare_splits(df):
    train = df[(df['eom'] >= '2005-01-01') & (df['eom'] <= '2015-12-31')]
    val = df[(df['eom'] >= '2016-01-01') & (df['eom'] <= '2018-12-31')]
    test = df[(df['eom'] >= '2019-01-01') & (df['eom'] <= '2024-12-31')]
    return train, val, test


def fit_preprocessor(train_df, features):
    train = train_df.copy()
    stats = {}
    for feat in features:
        if feat == 'me':
            train[feat] = np.log(train[feat].clip(lower=1e-6))
        train[feat] = pd.to_numeric(train[feat], errors='coerce')
        lower = train[feat].quantile(0.01)
        upper = train[feat].quantile(0.99)
        median = train[feat].median()
        stats[feat] = {'lower': lower, 'upper': upper, 'median': median}
        train[feat] = train[feat].clip(lower, upper).fillna(median)
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    stats['scaler'] = scaler
    return train, stats


def apply_preprocessor(df, features, stats):
    out = df.copy()
    for feat in features:
        if feat == 'me':
            out[feat] = np.log(out[feat].clip(lower=1e-6))
        out[feat] = pd.to_numeric(out[feat], errors='coerce')
        out[feat] = out[feat].clip(stats[feat]['lower'], stats[feat]['upper']).fillna(stats[feat]['median'])
    out[features] = stats['scaler'].transform(out[features])
    return out


def fit_models(X_train, y_train, X_val, y_val):
    ols = LinearRegression()
    ols.fit(X_train, y_train)

    best_alpha = None
    best_r2 = -np.inf
    for alpha in [0.01, 0.1, 1, 10, 100]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        r2 = r2_score(y_val, ridge.predict(X_val))
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)
    return ols, ridge, best_alpha


def evaluate_model(model, X, target, name, id_series, stock_means, global_mean):
    if name == 'Historical Average':
        preds = id_series.map(stock_means).fillna(global_mean)
    else:
        preds = model.predict(X)
    r2 = r2_score(target, preds)
    return r2, preds


def portfolio_returns(test_df, predictions, target_col='ret_exc_lead1m'):
    test_df = test_df.copy()
    for name, preds in predictions.items():
        test_df[f'pred_{name}'] = preds

    rows = []
    for date, month_df in test_df.groupby('eom'):
        for name in predictions:
            pred_col = f'pred_{name}'
            pos = month_df[month_df[pred_col] > 0]
            neg = month_df[month_df[pred_col] < 0]
            if len(pos) == 0 and len(neg) == 0:
                rows.append({'eom': date, 'model': name, 'return': 0.0})
                continue
            long_ret = pos[target_col].mean() if len(pos) > 0 else 0.0
            short_ret = neg[target_col].mean() if len(neg) > 0 else 0.0
            ret = (long_ret if len(pos) > 0 else 0.0) - (short_ret if len(neg) > 0 else 0.0)
            rows.append({'eom': date, 'model': name, 'return': ret})
    return pd.DataFrame(rows)


def calc_metrics(returns):
    mean_ret = returns.mean() * 12
    vol = returns.std(ddof=0) * np.sqrt(12)
    sharpe = mean_ret / vol if vol > 0 else 0.0
    return mean_ret, vol, sharpe


def plot_results(results):
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), [v['OOS R2'] for v in results.values()])
    plt.title('Out-of-Sample R² Comparison')
    plt.ylabel('R²')
    plt.savefig('r2_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        df = res['portfolio'].sort_values('eom')
        cum = (1 + df['return']).cumprod()
        plt.plot(df['eom'], cum, label=name)
    plt.title('Cumulative Portfolio Returns')
    plt.legend()
    plt.savefig('cumulative_returns.png')
    plt.close()


def main():
    args = parse_args()
    df = load_data(args.data_path)
    print('Data columns:', df.columns.tolist())
    print('Sample rows:')
    print(df[['id', 'eom', 'excntry', 'ret_exc_lead1m']].head().to_string(index=False))

    features = build_feature_list(df)
    print('Using features count:', len(features))

    train_df, val_df, test_df = prepare_splits(df)
    train_df = train_df[train_df['ret_exc_lead1m'].notna()].copy()
    val_df = val_df[val_df['ret_exc_lead1m'].notna()].copy()
    test_df = test_df[test_df['ret_exc_lead1m'].notna()].copy()
    print('Train/Val/Test rows after dropping NaN targets:', len(train_df), len(val_df), len(test_df))

    train_df, stats = fit_preprocessor(train_df, features)
    val_df = apply_preprocessor(val_df, features, stats)
    test_df = apply_preprocessor(test_df, features, stats)

    stock_means = train_df.groupby('id')['ret_exc_lead1m'].mean()
    global_mean = train_df['ret_exc_lead1m'].mean()

    ols, ridge, best_alpha = fit_models(
        train_df[features], train_df['ret_exc_lead1m'],
        val_df[features], val_df['ret_exc_lead1m'],
    )
    print('Best Ridge alpha:', best_alpha)

    models = {
        'Historical Average': None,
        'OLS': ols,
        'Ridge': ridge,
    }

    results = {}
    for name, model in models.items():
        r2_test, preds_test = evaluate_model(
            model,
            test_df[features] if model is not None else None,
            test_df['ret_exc_lead1m'],
            name,
            test_df['id'],
            stock_means,
            global_mean,
        )
        portfolio = portfolio_returns(test_df, {name: preds_test})
        results[name] = {
            'OOS R2': r2_test,
            'preds': preds_test,
            'portfolio': portfolio,
        }
        print(f'{name} OOS R²: {r2_test:.6f}')

    for name, res in results.items():
        mean_ret, vol, sharpe = calc_metrics(res['portfolio']['return'])
        print(f'{name} portfolio annualized mean={mean_ret:.6f}, vol={vol:.6f}, sharpe={sharpe:.6f}')

    plot_results(results)
    print('Saved plots: r2_comparison.png, cumulative_returns.png')


if __name__ == '__main__':
    main()
