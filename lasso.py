import pandas as pd
import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cormtx
import preprocess
import os
from pathlib import Path
from tqdm.auto import tqdm
from asgl import Regressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import AgglomerativeClustering
import threadpoolctl

# load data
def main(data_path, solver : str = 'CLARABEL', penalization : str = 'asgl') :
    cfg = preprocess.PreprocessConfig(
        input_path=Path(data_path),
        output_root=Path("../split_data")
    )
    train_df, val_df, test_df = preprocess.transform(
        *preprocess.split(preprocess.load_data(cfg), cfg), 
        cfg,
        exclude=['excntry'])
    
    print(f"train: {len(train_df)} rows")
    print(f"val:   {len(val_df)} rows")
    print(f"test:  {len(test_df)} rows")

    print("Computing correlation matrix...")

    cor_mtx = cormtx.association_matrix(train_df, datetime_cols=[cfg.date_col], categorical_cols=['excntry'])

    cormtx.plot_clustermap(cor_mtx, file_name=os.path.basename(data_path))

    opt_lambdas = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    opt_num_cluster = np.arange(5, 11, 1)

    logger = tqdm(total=len(opt_lambdas) * len(opt_num_cluster), desc="Tuning AGL")

    bst_score, bst_lambda, bst_grp_idx = -np.inf, None, None

    X_train, y_train = preprocess.get_xy(train_df, cfg)
    X_val, y_val = preprocess.get_xy(val_df, cfg)
    X_test, y_test = preprocess.get_xy(test_df, cfg)

    X_train, X_val, X_test = preprocess.onehot(X_train, X_val, X_test, 'excntry')

    for num_clus in opt_num_cluster :
        clus = AgglomerativeClustering(n_clusters=num_clus, linkage='average', metric='correlation')
        clus.fit(cor_mtx)
        grp_series = pd.Series(clus.labels_, index=train_df.columns)

        grp_idx = grp_series.reindex(X_train.columns)
        if grp_idx.isna().any():
            excntry_group = grp_series.loc['excntry']
            ohe_cols = [col for col in X_train.columns if col.startswith('excntry_')]
            grp_idx.loc[ohe_cols] = excntry_group
        grp_idx = grp_idx.astype(int)

        for opt in opt_lambdas :
            model = Regressor(penalization=penalization, solver=solver, lambda1=opt)
            model.fit(X_train, y_train, group_index=grp_idx)
            val_score = model.score(X_val, y_val)
            if val_score > bst_score :
                bst_score, bst_lambda, bst_grp_idx = val_score, opt, grp_idx

            logger.update(1)
    # use the best lambda to retrain on train+val
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_agl = Regressor(penalization=penalization, solver=solver, lambda1=bst_lambda)

    final_agl.fit(X_trainval, y_trainval, group_index=bst_grp_idx)
    score_agl = final_agl.score(X_test, y_test)

    ols = LinearRegression()
    ols.fit(X_trainval, y_trainval)
    score_ols = ols.score(X_test, y_test)

    bst_ridge_lambda, bst_ridge_score = None, -np.inf
    for lambda_ in tqdm(opt_lambdas, desc="Tuning Ridge") :
        ridge = Ridge(alpha=lambda_)
        ridge.fit(X_trainval, y_trainval)
        score = ridge.score(X_test, y_test)
        if score > bst_ridge_score :
            bst_ridge_lambda, bst_ridge_score = lambda_, score
    
    bst_lasso_lambda, bst_lasso_score = None, -np.inf
    for lambda_ in tqdm(opt_lambdas, desc="Tuning Lasso") :
        lasso = Lasso(alpha=lambda_, max_iter=10000)
        lasso.fit(X_trainval, y_trainval)
        score = lasso.score(X_test, y_test)
        if score > bst_lasso_score :
            bst_lasso_lambda, bst_lasso_score = lambda_, score
    
    log_txts = [
                f"Best lambda: {bst_lambda}", 
                f"Best group index:\n{bst_grp_idx}",
                f"Test R^2 of AGL: {score_agl:.6f}",
                f"Test R^2 of OLS: {score_ols:.6f}",
                f"Best lambda for Ridge: {bst_ridge_lambda}, Test R^2: {bst_ridge_score:.6f}",
                f"Best lambda for Lasso: {bst_lasso_lambda}, Test R^2: {bst_lasso_score:.6f}"]
    log_txt = "\n".join(log_txts)
    print(log_txt)
    with open(f"results_{os.path.basename(data_path).replace('.parquet', '.txt')}", "w") as f :
        f.write(log_txt)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./jkp_data_slim.parquet")
    parser.add_argument("--penalization", default="asgl")
    parser.add_argument("--solver", default="CLARABEL")
    args = parser.parse_args()
    main(args.data_path, args.solver, args.penalization)