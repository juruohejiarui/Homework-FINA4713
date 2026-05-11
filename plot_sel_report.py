import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path (default: output.png)")
    args = parser.parse_args()

    # 读取数据，第一列作为特征名
    df = pd.read_csv(args.input, index_col=0)

    # 设置全局绘图样式
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300

    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Feature Analysis Dashboard", fontsize=18, fontweight='bold')

    # 1. 选择频率 sel_freq (降序)
    sorted_freq = df["sel_freq"].sort_values(ascending=False)
    ax1 = axes[0, 0]
    sns.barplot(x=sorted_freq.values, y=sorted_freq.index, palette="coolwarm", ax=ax1)
    ax1.set_title("Feature Selection Frequency")
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("Feature")

    # 2. IC均值与标准差 (误差条)
    ax2 = axes[0, 1]
    # 按 ic_mean 排序
    sorted_ic = df.sort_values("ic_mean")
    y_pos = range(len(sorted_ic))
    ax2.errorbar(sorted_ic["ic_mean"], y_pos, xerr=sorted_ic["ic_std"], fmt='o', 
                 ecolor='gray', elinewidth=1, capsize=3, markersize=5, color='steelblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_ic.index, fontsize=8)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_title("Information Coefficient (Mean ± Std)")
    ax2.set_xlabel("IC Mean")
    ax2.set_ylabel("Feature")

    # 3. t 统计量，按符号着色
    ax3 = axes[1, 0]
    colors = df["dominant_sign_label"].map({"+": "green", "-": "red"})
    sorted_t = df.sort_values("t_stat")
    ax3.barh(sorted_t.index, sorted_t["t_stat"], color=colors.loc[sorted_t.index])
    ax3.axvline(0, color='black', linewidth=1)
    ax3.set_title("t-statistic (green: pos, red: neg)")
    ax3.set_xlabel("t-statistic")
    ax3.set_ylabel("Feature")

    # 4. 正负样本数量堆叠条
    ax4 = axes[1, 1]
    df_sorted = df.sort_values("sel_freq", ascending=False)  # 或其他排序
    ax4.barh(df_sorted.index, df_sorted["n_pos"], label="n_pos", color="dodgerblue")
    ax4.barh(df_sorted.index, df_sorted["n_neg"], left=df_sorted["n_pos"], label="n_neg", color="tomato")
    ax4.barh(df_sorted.index, df_sorted["n_zero"], left=df_sorted["n_pos"] + df_sorted["n_neg"], 
             label="n_zero", color="lightgray")
    ax4.set_title("Positive / Negative / Zero counts")
    ax4.set_xlabel("Count")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()