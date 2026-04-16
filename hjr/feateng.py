import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import shared_memory, get_context
from concurrent.futures import ProcessPoolExecutor

def load_data(data_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(data_path)
    
    # split data set
    train = df[(df['eom'] >= '2005-01-01') & (df['eom'] <= '2015-12-31')]
    val = df[(df['eom'] >= '2016-01-01') & (df['eom'] <= '2018-12-31')]
    test = df[(df['eom'] >= '2019-01-01') & (df['eom'] <= '2024-12-31')]

    train = train[train['ret_exc_lead1m'].notna()].copy()
    val = val[val['ret_exc_lead1m'].notna()].copy()
    test = test[test['ret_exc_lead1m'].notna()].copy()

    return (train, val, test)


# ---------------------------
# 1) 预处理：构造共享内存矩阵
# ---------------------------

def _is_categorical_col(s: pd.Series) -> bool:
    return (
        isinstance(s.dtype, pd.StringDtype)
        or isinstance(s.dtype, pd.BooleanDtype)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )

def build_shared_matrix(df, datetime_cols=None, categorical_cols=None):
    """
    把 DataFrame 预处理成一个共享的 float64 矩阵：
    - 数值列 / 日期列 -> float64
    - 类别列 -> factorize 编码后转 float64（缺失值为 NaN）

    返回：
      shm: SharedMemory 对象（父进程负责最后 unlink）
      shape: 矩阵形状
      columns: 列名顺序
      kinds: 每列类型，'num' 或 'cat'
      n_categories: 每个类别列的类别数；非类别列为 0
    """
    df = df.copy()
    datetime_cols = set(datetime_cols or [])
    categorical_cols = set(categorical_cols or [])

    # 自动补充类别列
    for col in df.columns:
        if col not in datetime_cols and col not in categorical_cols:
            if _is_categorical_col(df[col]):
                categorical_cols.add(col)

    cols = list(df.columns)
    n_rows, n_cols = df.shape
    data = np.empty((n_rows, n_cols), dtype=np.float64)

    kinds = []
    n_categories = []

    for j, col in enumerate(cols):
        s = df[col]

        if col in datetime_cols:
            dt = pd.to_datetime(s, errors="coerce")
            origin = dt.min()
            # 转成“天”为单位的连续浮点数；只要是线性变换，相关系数不受影响
            arr = ((dt - origin) / np.timedelta64(1, "D")).astype("float64")
            data[:, j] = arr.to_numpy()
            kinds.append("num")
            n_categories.append(0)

        elif col in categorical_cols:
            codes, uniques = pd.factorize(s, sort=False)
            arr = codes.astype(np.float64)
            arr[codes == -1] = np.nan
            data[:, j] = arr
            kinds.append("cat")
            n_categories.append(len(uniques))

        else:
            arr = pd.to_numeric(s, errors="coerce").astype("float64")
            data[:, j] = arr.to_numpy()
            kinds.append("num")
            n_categories.append(0)

    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_arr = np.ndarray(data.shape, dtype=np.float64, buffer=shm.buf)
    shm_arr[:] = data

    return shm, data.shape, cols, kinds, n_categories


# ---------------------------
# 2) 三种关联计算
# ---------------------------

def _pearson_num_num(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def _correlation_ratio(cat_codes, values, n_cat):
    """
    类别 vs 数值
    """
    mask = np.isfinite(cat_codes) & np.isfinite(values)
    if mask.sum() == 0:
        return np.nan

    cats = cat_codes[mask].astype(np.int64)
    vals = values[mask]

    counts = np.bincount(cats, minlength=n_cat)
    sums = np.bincount(cats, weights=vals, minlength=n_cat)

    nonzero = counts > 0
    means = np.zeros(n_cat, dtype=np.float64)
    means[nonzero] = sums[nonzero] / counts[nonzero]

    grand_mean = vals.mean()
    ss_between = np.sum(counts[nonzero] * (means[nonzero] - grand_mean) ** 2)
    ss_total = np.sum((vals - grand_mean) ** 2)

    if ss_total == 0:
        return 0.0

    return float(np.sqrt(ss_between / ss_total))

def _cramers_v(x_codes, y_codes, nx, ny):
    """
    类别 vs 类别
    """
    mask = np.isfinite(x_codes) & np.isfinite(y_codes)
    if mask.sum() == 0:
        return np.nan

    x = x_codes[mask].astype(np.int64)
    y = y_codes[mask].astype(np.int64)

    table = np.zeros((nx, ny), dtype=np.int64)
    np.add.at(table, (x, y), 1)

    n = table.sum()
    if n == 0:
        return np.nan

    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n

    valid = expected > 0
    chi2 = np.sum(((table - expected) ** 2)[valid] / expected[valid])

    phi2 = chi2 / n
    denom = min(nx - 1, ny - 1)
    if denom <= 0:
        return 0.0

    return float(np.sqrt(phi2 / denom))


# ---------------------------
# 3) 子进程：只通过列索引读取共享内存
# ---------------------------

_G_ARR = None
_G_SHM = None
_G_KINDS = None
_G_NCAT = None

def _worker_init(shm_name, shape, kinds, n_categories):
    global _G_ARR, _G_SHM, _G_KINDS, _G_NCAT
    _G_SHM = shared_memory.SharedMemory(name=shm_name)
    _G_ARR = np.ndarray(shape, dtype=np.float64, buffer=_G_SHM.buf)
    _G_KINDS = kinds
    _G_NCAT = n_categories

def _pair_task(pair):
    i, j = pair
    x = _G_ARR[:, i]
    y = _G_ARR[:, j]

    kind_i = _G_KINDS[i]
    kind_j = _G_KINDS[j]

    if kind_i == "cat" and kind_j == "cat":
        v = _cramers_v(x, y, _G_NCAT[i], _G_NCAT[j])
    elif kind_i == "cat" and kind_j == "num":
        v = _correlation_ratio(x, y, _G_NCAT[i])
    elif kind_i == "num" and kind_j == "cat":
        v = _correlation_ratio(y, x, _G_NCAT[j])
    else:
        v = _pearson_num_num(x, y)

    return i, j, v


# ---------------------------
# 4) 主函数：8 进程计算矩阵
# ---------------------------

def mixed_association_matrix_parallel(df, datetime_cols=None, categorical_cols=None, n_jobs=8):
    """
    返回一个对称矩阵：
    - 数值/日期 vs 数值/日期：Pearson
    - 类别 vs 数值：Correlation Ratio
    - 类别 vs 类别：Cramér's V
    """
    shm, shape, cols, kinds, n_categories = build_shared_matrix(
        df,
        datetime_cols=datetime_cols,
        categorical_cols=categorical_cols,
    )

    try:
        n = len(cols)
        mat = pd.DataFrame(np.eye(n, dtype=np.float64), index=cols, columns=cols)

        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        ctx = get_context("spawn")  # 更稳妥，跨平台；Linux 下也可用 fork
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(shm.name, shape, kinds, n_categories),
        ) as ex:
            # chunksize 可以减少调度开销
            for i, j, v in tqdm(ex.map(_pair_task, pairs, chunksize=16)):
                mat.iat[i, j] = v
                mat.iat[j, i] = v

        return mat

    finally:
        shm.close()
        shm.unlink()

def plot_clustermap(matrix, figsize=(15, 15), use_abs_for_clustering=True, cmap="coolwarm"):
    m = matrix.copy()

    if use_abs_for_clustering:
        cluster_data = m.abs()
    else:
        cluster_data = m.copy()

    # np.fill_diagonal(cluster_data.values, 1.0)

    # clustermap 会自动对行列做聚类
    g = sns.clustermap(
        cluster_data, 
        row_cluster=True,
        col_cluster=True,
        row_linkage=None,
        col_linkage=None,
        cmap=cmap,
        center=0 if (m.values.min() < 0) else None,
        figsize=figsize,
        linewidths=0.2,
    )
    plt.savefig("./cor_matrix.png")

# load data
def main(data_path) :
    train_df, val_df, test_df = load_data(data_path)
    
    print(f"train: {len(train_df)} rows")
    print(f"val:   {len(val_df)} rows")
    print(f"test:  {len(test_df)} rows")

    mtx = mixed_association_matrix_parallel(train_df.drop(columns=['id']))

    plot_clustermap(mtx)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../jkp_data.parquet")
    args = parser.parse_args()
    main(args.data_path)