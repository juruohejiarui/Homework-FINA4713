import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cormtx
import preprocess
import os
from pathlib import Path
from asgl import Regressor
from sklearn.cluster import AgglomerativeClustering

def grp_feats(cor_mtx) :
    cluster = AgglomerativeClustering(n_clusters=30)
    
    pass

# load data
def main(data_path) :
    cfg = preprocess.PreprocessConfig(
        input_path=Path(data_path),
        output_root=Path("../split_data")
    )
    train_df, val_df, test_df = preprocess.load_data(cfg)
    
    print(f"train: {len(train_df)} rows")
    print(f"val:   {len(val_df)} rows")
    print(f"test:  {len(test_df)} rows")

    corr_mtx = cormtx.mixed_association_matrix_parallel(train_df.drop(columns=['id']))

    # cluster for group idx
    grp_info = cormtx.plot_clustermap(corr_mtx, os.path.basename(data_path))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./jkp_data_slim.parquet")
    args = parser.parse_args()
    main(args.data_path)