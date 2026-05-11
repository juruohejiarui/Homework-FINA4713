import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot cumulative returns with Sharpe ratios.")
    parser.add_argument("pf_file", help="Path to portfolio returns CSV (with date column)")
    parser.add_argument("sharpe_file", help="Path to Sharpe ratio CSV")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
    args = parser.parse_args()

    # 读取投资组合收益数据
    pf_df = pd.read_csv(args.pf_file, parse_dates=["date"], index_col="date")
    pf_df.sort_index(inplace=True)

    # 读取夏普比率数据，第一列应为策略名称，需与 pf_df 列名对应
    sharpe_df = pd.read_csv(args.sharpe_file, index_col=0)

    # 计算累计收益曲线：cumulative = (1 + r).cumprod() - 1
    cum_ret = (1 + pf_df).cumprod() - 1

    plt.figure(figsize=(12, 6))
    # 为每个列绘制曲线
    for col in cum_ret.columns:
        if col == "market":
            plt.plot(cum_ret.index, cum_ret[col], linestyle="--", linewidth=2, label="market", color="black")
        else:
            # 查找对应的夏普比率
            sr = sharpe_df.loc[col, "sharpe"] if col in sharpe_df.index else np.nan
            label = f"{col} (Sharpe={sr:.2f})" if not np.isnan(sr) else col
            plt.plot(cum_ret.index, cum_ret[col], linewidth=1.5, label=label)

    # 设置x轴时间刻度，显示年份
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    plt.title("Cumulative Returns of Portfolio Strategies")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Return")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Figure saved to {args.output}")

if __name__ == "__main__":
    main()