import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt

# 读取 CSV 文件，第一列作为行索引（即特征名）

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("input_csv", type=str, default="same_group_freq_pca-tr6-tr3-step3", help="Path to the input CSV file containing the same-group frequency matrix.")
argument_parser.add_argument("--output_png", type=str, default="output.png", help="Path to save the output heatmap PNG file.")
args = argument_parser.parse_args()

df = pd.read_csv(args.input_csv, index_col=0)

# 设置图形大小
plt.figure(figsize=(14, 12))

# 绘制热力图
sns.heatmap(
    df,
    annot=False,            # 如果数据值差异不大可设为 True 显示数值
    cmap="coolwarm",        # 颜色映射，也可选 "RdBu_r", "viridis" 等
    square=True,            # 单元格呈正方形
    linewidths=0.5,         # 单元格之间细线
    xticklabels=True,       # 显示列名
    yticklabels=True,       # 显示行名
    cbar_kws={"shrink": 0.8}  # 调整颜色条大小
)

plt.xticks(rotation=90)     # 列名旋转，避免重叠
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(args.output_png, dpi=300)