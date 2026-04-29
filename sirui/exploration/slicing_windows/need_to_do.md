## 项目背景
我正在做 FINA 4713 课程的 Final Project。任务是使用 Jensen, Kelly, and Pedersen (2023) 数据集，构建月度股票收益预测模型（横截面预测）。

数据是面板结构：每个时间点 t（月份），有大量股票，每只股票有多个特征（预测因子）。我的目标是：用 t 时刻的特征，预测 t+1 时刻的个股收益。

## 我要完成的任务
使用 **滚动窗口 + 嵌入式特征选择** 的方法，自动筛选稳健的特征子集。具体流程如下：

### Step 1: 滚动窗口划分数据
- 数据集时间范围：假设从 2000-01 到 2020-12（具体以实际数据为准）
- 滚动窗口设置：
  - 训练窗口大小：132 个月（11年）
  - 验证/测试窗口大小：12 个月（1年）
  - 滚动步长：12 个月（每年重新训练一次特征选择模型）
- 每个窗口内的数据划分：
  - 训练集：窗口内前 80% 时间
  - 验证集：窗口内后 20% 时间（用于特征选择的评估）

### Step 2: 每个窗口内，使用 Featurewiz 进行自动特征选择
- 对每个滚动窗口的训练数据，运行 `featurewiz` 库
- 配置：`featurewiz.FeatureWiz(verbose=1)`，使用默认的 XGBoost 作为底层模型
- 记录该窗口选中的特征名称列表

### Step 3: 特征稳定性筛选
- 假设总共有 N 个滚动窗口
- 每个特征被选中的次数 = count_i
- 保留被选中频率 > 某个阈值（如 70%）的特征，作为最终的"稳健特征集"
- 阈值建议可调：先试试 0.7，也可以尝试 0.5 和 0.9 做敏感性分析

### Step 4: 输出和可视化
- 打印每个特征的选中频率（按降序排列）
- 绘制条形图展示"特征选中频率分布"
- 输出最终的稳健特征列表

## 代码要求
请用 Python 编写完整代码，包含：

1. **必要的 import 语句**：pandas, numpy, featurewiz, sklearn.model_selection, matplotlib 等
2. **函数封装**：
   - `rolling_window_split(data, train_size=120, val_size=12, step_size=12)`：生成时间窗口索引
   - `feature_selection_in_window(X_train, y_train)`：在一个窗口内运行 featurewiz，返回选中特征
   - `aggregate_feature_stability(all_selected_features, threshold=0.7)`：汇总并筛选稳定特征
3. **主循环**：遍历所有滚动窗口，执行上述流程
4. **代码注释**：关键步骤添加注释，说明为什么这样做（特别是时间序列数据的特殊处理）
5. **进度显示**：使用 tqdm 显示循环进度

## 重要注意事项
- 时间序列数据不能用随机 shuffle！训练集必须按时间顺序在验证集之前
- 特征选择时只用训练集（X_train, y_train），绝不能用验证集或未来的数据
- y (target) 是 t+1 期的个股收益率
- Featurewiz 内部需要验证集，请用 `fit_transform` 时传入 `X_train, y_train` 即可
- 如果某个窗口的特征选择失败（如 featurewiz 报错），跳过该窗口并给出警告

## 假设数据格式
假设数据已经加载为：
- `df`：DataFrame，列包括 ['date', 'stock_id', 'return_future', 'feature1', 'feature2', ...]
- `features`：特征列名列表（不包含 date, stock_id, return_future）
- `target_col = 'return_future'`