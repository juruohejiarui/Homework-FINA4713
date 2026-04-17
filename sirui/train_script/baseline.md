# baseline.py 说明

这个脚本是 FINA4713 项目的 baseline 实验主流程，对应作业要求中的 **3.3 Model estimation**、**3.4 Out-of-sample evaluation**、**3.5 Portfolio construction**。

---

## 我做了什么

`baseline.py` 完成以下工作：

1. 读取预处理后的训练/验证/测试集（parquet）。
2. 训练并评估三个模型：
   - `historical_average`：每只股票用训练期历史平均收益预测；
   - `ols`：普通最小二乘线性回归；
   - `ridge`：机器学习基线（L2 正则线性模型）。
3. 用验证集选择 Ridge 的超参数 `alpha`（在给定网格上选验证集 MSE 最小）。
4. 在测试集计算每个模型的 out-of-sample `R^2`。
5. 每个测试月将预测值转成组合权重，计算组合月收益，并汇总：
   - 年化超额收益；
   - 年化波动率；
   - 年化 Sharpe。
6. 输出结果文件（指标、预测、组合收益序列、运行摘要）。

---

## 输入数据

默认读取：

- `data/train/train_processed.parquet`
- `data/val/val_processed.parquet`
- `data/test/test_processed.parquet`

默认关键列：

- `id`：股票标识
- `eom`：月份日期
- `ret_exc_lead1m`：下月超额收益（预测目标）

其余列视为特征。

---

## 运行方式

在项目根目录执行：

```bash
python "f:/Files/code/2026_spring_term/FINA4713/Project/train_script/baseline.py"
```

可选参数示例（修改 Ridge 网格）：

```bash
python "f:/Files/code/2026_spring_term/FINA4713/Project/train_script/baseline.py" --ridge_alphas "0.001,0.01,0.1,1,10,100"
```

---

## 输出文件

默认输出目录：`output/baseline`

- `model_oos_metrics.csv`：三个模型的验证集 MSE 和测试集 OOS `R^2`
- `portfolio_metrics.csv`：三个模型对应组合的年化收益/年化波动/年化 Sharpe
- `test_predictions.csv`：测试集逐股票逐月预测值
- `portfolio_monthly_returns_historical_average.csv`
- `portfolio_monthly_returns_ols.csv`
- `portfolio_monthly_returns_ridge.csv`
- `run_summary.json`：运行参数、最优超参数、结果总览、解释模板

---

## 方法细节（简要）

### 1) OOS R^2 定义

测试集 OOS `R^2` 按下式计算：

`OOS R^2 = 1 - SSE_model / SSE_null`

其中 `SSE_null` 的基准预测是训练集目标均值。

### 2) 组合权重构造

对每个月横截面：

- 先将预测值去均值；
- 再按 `sum(abs(w)) = 1` 归一化。

这对应一个资金中性的 long-short 组合（多空权重和约为 0）。

---

## 和作业要求的对应关系

- **3.3 Model estimation**：已包含历史均值、OLS、Ridge 三个模型，且 Ridge 用验证集选超参数。
- **3.4 Out-of-sample evaluation**：已输出测试集 OOS `R^2`。
- **3.5 Portfolio construction**：已按月构建权重并输出年化收益、波动、Sharpe。

---

## 注意事项

- 如果测试集中出现训练集中没有的 `id`，历史均值模型会回退到训练集全局均值。
- 月频收益预测噪声很大，出现负的 OOS `R^2` 并不罕见。
- 当前组合评估未显式扣除交易成本；报告中应补充对换手、流动性、冲击成本的定性讨论。
