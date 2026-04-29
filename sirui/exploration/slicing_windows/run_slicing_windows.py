"""
滚动窗口 + FeatureWiz 嵌入式特征选择，汇总跨窗特征稳定性。

输出 CSV 除 selection_frequency 外，含符号稳定性：各窗训练段内月均 Rank IC 的正负是否一致
（sign_stability、dominant_sign_label、mean_train_ic_across_windows 等）。

时间序列注意：窗口按日历月末顺序滑动，训练段严格早于验证段，不做随机打乱。
数据范围：默认仅用 train parquet（不含 test）。默认 **train_size=12、val_size=0、step_size=12**，即每个滚动窗仅为连续 **12** 个唯一月末，
FeatureWiz 与 Rank IC 均只用该窗内这 12 个月；课程 train 约 **132** 个月时约 **11** 个窗（(132-12)/12+1）。需要更长训练窗（如 120+12）请显式改 CLI，
并注意仅用 train 日历时大块窗往往只能滚出极少数窗；亦可 `--concat-valid-path` 拼 valid（仍不用 test）。

用法:
  python run_slicing_windows.py
  python run_slicing_windows.py --thresholds 0.5 0.7 0.9 --featurewiz-nrows 80000
  python run_slicing_windows.py --min-robust-selection-freq 0.7 --min-robust-sign-stability 0.9

已有 CSV 时仅筛选打印（PowerShell 一行）:
  python -c "import pandas as pd; df=pd.read_csv(r'output/feature_selection_frequency.csv'); r=df.query('selection_frequency>=0.7 & sign_stability>=0.85 & n_windows_selected>=5'); print(r.sort_values(['sign_stability','selection_frequency'],ascending=False).to_string(index=False))"
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI：只保存 PNG，不弹窗阻塞
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # noqa: F401
from tqdm import tqdm

# TimeSeriesSplit 用于「按时间顺序」的样本折；本脚本为面板月末滚动，窗口由 rolling_window_split 显式生成。

_BASELINE = Path(__file__).resolve().parent.parent.parent / "baseline"
sys.path.insert(0, str(_BASELINE))
import baseline as bl  # noqa: E402

_DEFAULT_TRAIN = "f:/Files/code/2026_spring_term/FINA4713/Project/data/train/train_processed.parquet"
_DEFAULT_VALID = "f:/Files/code/2026_spring_term/FINA4713/Project/data/val/val_processed.parquet"

# 仅用 train（约 132 个唯一月末）：单窗跨度 = train_size + val_size = 12 → (132−12)/12+1 ≈ 11 窗。
_DEFAULT_ROLL_TRAIN_MONTHS = 12
_DEFAULT_ROLL_VAL_MONTHS = 0
_DEFAULT_ROLL_STEP_MONTHS = 12


def rolling_window_split(
    data: pd.DataFrame,
    train_size: int = _DEFAULT_ROLL_TRAIN_MONTHS,
    val_size: int = _DEFAULT_ROLL_VAL_MONTHS,
    step_size: int = _DEFAULT_ROLL_STEP_MONTHS,
    *,
    date_col: str = "eom",
) -> list[dict]:
    """
    生成按时间顺序的滚动窗口索引（仅使用唯一月末，避免 shuffle）。

    每个窗口连续包含 train_size + val_size 个月：
    - 前 train_size 个月：用于特征选择（仅该段送入 FeatureWiz，符合「只用训练集」）
    - 后 val_size 个月：可选占位验证段（可为 0；主流程不用于 fit）

    窗口每次向前滑动 step_size 个月。
    """
    dates = np.sort(data[date_col].unique())
    n = len(dates)
    span = train_size + val_size
    windows: list[dict] = []
    start = 0
    idx = 0
    while start + span <= n:
        block = dates[start : start + span]
        train_dates = block[:train_size]
        val_dates = block[train_size:]
        windows.append(
            {
                "i_window": idx,
                "start": pd.Timestamp(train_dates[0]),
                "end": pd.Timestamp(block[-1]),
                "train_dates": train_dates,
                "val_dates": val_dates,
            }
        )
        idx += 1
        start += step_size
    return windows


def feature_selection_in_window(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    verbose: int = 1,
    featurewiz_nrows: int | None = None,
) -> list[str]:
    """
    在单个窗口的训练段上运行 FeatureWiz（默认底层 XGBoost），返回选中特征名。
    """
    from featurewiz import FeatureWiz

    # 与课程说明一致：默认 FeatureWiz；nrows 可加速/控内存
    kwargs: dict = {"verbose": verbose}
    if featurewiz_nrows is not None:
        kwargs["nrows"] = int(featurewiz_nrows)

    fwiz = FeatureWiz(**kwargs)
    fwiz.fit_transform(X_train, y_train)
    feats = fwiz.features
    if feats is None:
        return []
    return [str(x) for x in feats]


def train_window_mean_rank_ic(
    tr: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    min_cs: int,
) -> dict[str, float]:
    """
    在当前窗口训练段内：每月 Spearman(特征, 目标)，再对月份取平均。
    用于符号稳定性（正/负 IC 是否跨窗一致）；仅用训练截面，不泄漏未来。
    """
    out: dict[str, float] = {}
    for f in features:
        if f not in tr.columns:
            continue
        ic_list: list[float] = []
        for _, g in tr.groupby(date_col, sort=True):
            if len(g) < min_cs:
                continue
            ic = g[f].corr(g[target_col], method="spearman")
            if ic == ic:
                ic_list.append(float(ic))
        out[f] = float(np.mean(ic_list)) if ic_list else float("nan")
    return out


def format_feature_table(df: pd.DataFrame, *, col_space: int = 2) -> str:
    """终端对齐输出：百分比、小数位统一。"""
    if df.empty:
        return "(空表)"
    disp = df.copy()
    pct_cols = {"selection_frequency", "sign_stability"}
    four_dec = {
        "mean_train_ic_across_windows",
        "std_train_ic_across_windows",
    }
    for c in disp.columns:
        if c in pct_cols:
            disp[c] = disp[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
        elif c in four_dec:
            disp[c] = disp[c].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "")
        elif np.issubdtype(disp[c].dtype, np.floating):
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return disp.to_string(index=False, col_space=col_space)


def filter_robust_freq_and_sign(
    freq_df: pd.DataFrame,
    *,
    min_selection_frequency: float,
    min_sign_stability: float,
    min_windows_selected: int,
) -> pd.DataFrame:
    """又常入选、又符号稳：入选频率 + sign_stability 阈值 + 至少入选够多窗。"""
    if freq_df.empty:
        return freq_df
    ss = freq_df["sign_stability"]
    m = (
        (freq_df["selection_frequency"] >= min_selection_frequency)
        & ss.notna()
        & (ss >= min_sign_stability)
        & (freq_df["n_windows_selected"] >= min_windows_selected)
    )
    out = freq_df.loc[m].copy()
    return out.sort_values(
        ["sign_stability", "selection_frequency", "mean_train_ic_across_windows"],
        ascending=[False, False, False],
    )


def summarize_sign_stability(
    ic_per_window_by_feature: dict[str, list[float]],
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    对每个特征，仅在「该窗被 FeatureWiz 选中」的窗口上收集 mean IC。
    sign_stability = max(n_pos, n_neg) / (n_pos + n_neg)，仅统计 IC 明确非零的窗口。
    """
    rows: list[dict] = []
    for feat, ms in ic_per_window_by_feature.items():
        n_sel = len(ms)
        ms_fin = [m for m in ms if m == m]
        if not ms_fin:
            rows.append(
                {
                    "feature": feat,
                    "n_windows_selected": n_sel,
                    "mean_train_ic_across_windows": float("nan"),
                    "std_train_ic_across_windows": float("nan"),
                    "n_windows_ic_positive": 0,
                    "n_windows_ic_negative": 0,
                    "n_windows_ic_near_zero": 0,
                    "sign_stability": float("nan"),
                    "dominant_sign": 0,
                    "dominant_sign_label": "",
                }
            )
            continue
        n_pos = sum(1 for m in ms_fin if m > eps)
        n_neg = sum(1 for m in ms_fin if m < -eps)
        n_zero = sum(1 for m in ms_fin if abs(m) <= eps)
        denom = n_pos + n_neg
        stab = float(max(n_pos, n_neg) / denom) if denom > 0 else float("nan")
        if n_pos > n_neg:
            dom, label = 1, "+"
        elif n_neg > n_pos:
            dom, label = -1, "-"
        else:
            dom, label = 0, "tie"
        rows.append(
            {
                "feature": feat,
                "n_windows_selected": n_sel,
                "mean_train_ic_across_windows": float(np.mean(ms_fin)),
                "std_train_ic_across_windows": float(np.std(ms_fin, ddof=1)) if len(ms_fin) > 1 else 0.0,
                "n_windows_ic_positive": int(n_pos),
                "n_windows_ic_negative": int(n_neg),
                "n_windows_ic_near_zero": int(n_zero),
                "sign_stability": stab,
                "dominant_sign": dom,
                "dominant_sign_label": label,
            }
        )
    return pd.DataFrame(rows)


def aggregate_feature_stability(
    all_selected_features: list[list[str]],
    threshold: float = 0.7,
) -> tuple[list[str], pd.Series]:
    """
    频率 = 该特征在多少个窗口中被选中 / N，N = 滚动窗口总数（含失败或空选集窗口，该窗对该特征贡献 0）。
    与 need_to_do 中「假设共 N 个窗口」一致。
    """
    n = len(all_selected_features)
    if n == 0:
        return [], pd.Series(dtype=float)

    counts: dict[str, int] = {}
    for feats in all_selected_features:
        for name in feats:
            counts[name] = counts.get(name, 0) + 1

    freq = pd.Series({k: v / n for k, v in counts.items()}).sort_values(ascending=False)
    stable = freq[freq > threshold].index.tolist()
    return stable, freq


def _load_full_panel(
    train_path: str,
    date_col: str,
    *,
    concat_valid_path: str | None = None,
) -> pd.DataFrame:
    train = bl.load_data(train_path, date_col)
    if concat_valid_path:
        valid = bl.load_data(concat_valid_path, date_col)
        full = pd.concat([train, valid], ignore_index=True)
    else:
        full = train
    return full.sort_values([date_col]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    out = Path(__file__).resolve().parent / "output"
    p = argparse.ArgumentParser(description="Rolling windows + FeatureWiz feature stability.")
    p.add_argument("--train_path", type=str, default=_DEFAULT_TRAIN)
    p.add_argument(
        "--concat-valid-path",
        type=str,
        default=None,
        metavar="PATH",
        help="若指定则在 train 后横向拼接该 parquet（一般为 valid），延长日历以得到多滚动窗；仍不使用 test。默认 None。",
    )
    p.add_argument("--output_dir", type=str, default=str(out))
    p.add_argument("--id_col", type=str, default="id")
    p.add_argument("--date_col", type=str, default="eom")
    p.add_argument("--target_col", type=str, default="ret_exc_lead1m")
    p.add_argument(
        "--train_size",
        type=int,
        default=_DEFAULT_ROLL_TRAIN_MONTHS,
        help="窗口内用于 FeatureWiz 的训练月数（默认 12：单窗恰为 12 个唯一月末）",
    )
    p.add_argument(
        "--val_size",
        type=int,
        default=_DEFAULT_ROLL_VAL_MONTHS,
        help="窗口内在训练月之后的占位月数，可不占长度（默认 0；总跨度 = train_size + val_size）",
    )
    p.add_argument(
        "--step_size",
        type=int,
        default=_DEFAULT_ROLL_STEP_MONTHS,
        help="滚动步长（月；默认 12）",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.5, 0.7, 0.9],
        help="稳健特征频率阈值（严格大于该值才保留）",
    )
    p.add_argument("--featurewiz-verbose", type=int, default=1)
    p.add_argument(
        "--featurewiz-nrows",
        type=int,
        default=None,
        help="传给 FeatureWiz 的 nrows（抽样行数，加速/省内存；None 表示全量）",
    )
    p.add_argument("--plot-top-n", type=int, default=40, help="条形图展示前 N 个特征")
    p.add_argument(
        "--min-cross-section-for-sign",
        type=int,
        default=30,
        help="计算训练窗内月均 Rank IC 时，每月至少多少只股票",
    )
    p.add_argument(
        "--min-robust-selection-freq",
        type=float,
        default=0.7,
        help="「又稳又常入选」筛选：最低选中频率",
    )
    p.add_argument(
        "--min-robust-sign-stability",
        type=float,
        default=0.85,
        help="「又稳又常入选」筛选：最低 sign_stability（IC 非零窗内方向一致度）",
    )
    p.add_argument(
        "--min-robust-windows",
        type=int,
        default=5,
        help="「又稳又常入选」筛选：至少入选多少个窗口",
    )
    p.add_argument("--print-rows", type=int, default=30, help="终端打印前 N 行全表")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = _load_full_panel(
        args.train_path,
        args.date_col,
        concat_valid_path=args.concat_valid_path,
    )
    feature_cols = bl.get_feature_cols(full, args.id_col, args.date_col, args.target_col)
    full, _ = bl.clean_for_modeling(full, feature_cols, args.target_col)

    n_months = int(full[args.date_col].nunique())
    span = int(args.train_size + args.val_size)
    windows = rolling_window_split(
        full,
        train_size=args.train_size,
        val_size=args.val_size,
        step_size=args.step_size,
        date_col=args.date_col,
    )
    if not windows:
        raise SystemExit("没有可生成的滚动窗口：请检查数据时间跨度与 train_size+val_size。")

    n_win = len(windows)
    if n_win < args.min_robust_windows:
        warnings.warn(
            (
                f"唯一月末={n_months}，单窗跨度(train_size+val_size)={span}，当前滚动窗数={n_win}，"
                f"小于 --min-robust-windows={args.min_robust_windows}，稳健筛选通常会为空。"
                "常见原因：仅用 train 时日历长度恰好等于窗口跨度，只能生成 1 窗。"
                "可选：在严守 test 前提下使用 --concat-valid-path 指向 valid parquet 延长日历；"
                "或缩小单窗跨度（默认 train_size=12,val_size=0,step_size=12）；"
                "或暂时降低 --min-robust-windows。"
            ),
            stacklevel=1,
        )

    all_selected: list[list[str]] = []
    meta_rows: list[dict] = []
    ic_per_window_by_feature: dict[str, list[float]] = defaultdict(list)

    for w in tqdm(windows, desc="rolling FeatureWiz"):
        train_mask = full[args.date_col].isin(w["train_dates"])
        tr = full.loc[train_mask]
        tr, _ = bl.clean_for_modeling(tr, feature_cols, args.target_col)
        if len(tr) < 1_000:
            warnings.warn(
                f"窗口 {w['i_window']} 训练行数过少 ({len(tr)})，跳过。",
                stacklevel=1,
            )
            all_selected.append([])
            meta_rows.append(
                {
                    "i_window": w["i_window"],
                    "start": str(w["start"]),
                    "end": str(w["end"]),
                    "n_train_rows": int(len(tr)),
                    "skipped": True,
                    "reason": "too_few_rows",
                }
            )
            continue

        X_tr = tr[feature_cols].copy()
        y_tr = tr[args.target_col]

        try:
            selected = feature_selection_in_window(
                X_tr,
                y_tr,
                verbose=args.featurewiz_verbose,
                featurewiz_nrows=args.featurewiz_nrows,
            )
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"窗口 {w['i_window']} FeatureWiz 失败，已跳过: {e}", stacklevel=1)
            all_selected.append([])
            meta_rows.append(
                {
                    "i_window": w["i_window"],
                    "start": str(w["start"]),
                    "end": str(w["end"]),
                    "n_train_rows": len(tr),
                    "skipped": True,
                    "error": repr(e),
                }
            )
            continue

        all_selected.append(selected)
        ic_means = train_window_mean_rank_ic(
            tr,
            selected,
            args.target_col,
            args.date_col,
            args.min_cross_section_for_sign,
        )
        for f in selected:
            ic_per_window_by_feature[f].append(ic_means.get(f, float("nan")))
        meta_rows.append(
            {
                "i_window": w["i_window"],
                "start": str(w["start"]),
                "end": str(w["end"]),
                "n_train_rows": len(tr),
                "n_selected": len(selected),
                "selected": selected,
                "train_mean_ic_selected": {k: ic_means[k] for k in selected if k in ic_means},
                "skipped": False,
            }
        )

    valid_n = sum(1 for s in all_selected if len(s) > 0)
    _, freq = aggregate_feature_stability(all_selected, threshold=0.0)

    freq_df = freq.reset_index()
    freq_df.columns = ["feature", "selection_frequency"]
    stab_df = summarize_sign_stability(dict(ic_per_window_by_feature))
    freq_df = freq_df.merge(stab_df, on="feature", how="left")
    freq_path = out_dir / "feature_selection_frequency.csv"
    freq_df.to_csv(freq_path, index=False)

    robust = filter_robust_freq_and_sign(
        freq_df,
        min_selection_frequency=args.min_robust_selection_freq,
        min_sign_stability=args.min_robust_sign_stability,
        min_windows_selected=args.min_robust_windows,
    )
    robust_path = out_dir / "feature_robust_freq_and_sign.csv"
    robust.to_csv(robust_path, index=False)

    stable_by_t: dict[str, list[str]] = {}
    for t in args.thresholds:
        stable, _ = aggregate_feature_stability(all_selected, threshold=t)
        stable_by_t[str(t)] = stable

    summary = {
        "n_unique_months_in_panel": n_months,
        "rolling_span_months": span,
        "concat_valid_path": args.concat_valid_path,
        "n_windows_total": len(windows),
        "n_windows_ran_featurewiz_ok": valid_n,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "step_size": args.step_size,
        "stable_features_by_threshold": stable_by_t,
        "note": (
            "selection_frequency = 选中次数 / 总窗口数。"
            "sign_stability = 在 mean IC 非零的入选窗中，与多数符号一致的占比；"
            "mean_train_ic_across_windows = 各窗训练段月均 Rank IC 的算术平均。"
        ),
        "min_cross_section_for_sign": args.min_cross_section_for_sign,
        "robust_filter": {
            "min_selection_frequency": args.min_robust_selection_freq,
            "min_sign_stability": args.min_robust_sign_stability,
            "min_windows_selected": args.min_robust_windows,
            "n_features_pass": int(len(robust)),
            "features": robust["feature"].tolist() if len(robust) else [],
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "windows": meta_rows}, f, ensure_ascii=True, indent=2, default=str)

    disp_cols = [
        "feature",
        "selection_frequency",
        "sign_stability",
        "dominant_sign_label",
        "mean_train_ic_across_windows",
        "n_windows_selected",
        "n_windows_ic_positive",
        "n_windows_ic_negative",
    ]
    disp_cols = [c for c in disp_cols if c in freq_df.columns]
    sorted_all = freq_df.sort_values("selection_frequency", ascending=False)
    n_show = max(1, args.print_rows)
    print(f"\n=== 特征选中频率与符号稳定性（按选中频率降序，前 {n_show} 行）===\n")
    print(format_feature_table(sorted_all.head(n_show)[disp_cols]))

    print(
        f"\n=== 又常入选又符号稳 "
        f"(freq≥{args.min_robust_selection_freq:.0%}, "
        f"sign_stability≥{args.min_robust_sign_stability:.0%}, "
        f"n_windows≥{args.min_robust_windows}) 共 {len(robust)} 个 ===\n"
    )
    if len(robust) == 0:
        print("(无满足条件的特征，可调低 --min-robust-* 参数)")
    else:
        print(format_feature_table(robust[disp_cols]))
    print(f"\n已写入: {robust_path}")

    print(f"\n各阈值下的稳健特征数量: {{{', '.join(f'{k}: {len(v)}' for k, v in stable_by_t.items())}}}")
    print(f"\n已保存: {freq_path}, {robust_path}, {out_dir / 'run_summary.json'}")

    # 条形图：选中频率分布
    if len(freq) == 0:
        print("无选中特征频率可绘制（可能 FeatureWiz 未安装或所有窗口失败），跳过条形图。")
    else:
        top = freq.head(max(1, args.plot_top_n))
        fig_h = max(4.0, 0.22 * len(top))
        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(top.index[::-1], top.values[::-1], color="steelblue", edgecolor="none")
        ax.set_xlabel("Selection frequency (share of all rolling windows)")
        ax.set_ylabel("Feature")
        ax.set_title("Feature selection frequency across rolling windows (FeatureWiz)")
        ax.set_xlim(0, 1.05)
        fig.tight_layout()
        fig_path = out_dir / "feature_frequency_barh.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存图: {fig_path}")


if __name__ == "__main__":
    main()
