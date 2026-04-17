# preprocess.py 处理流程说明

本文档说明 `data_preprocess/preprocess.py` 对 JKP 数据做了哪些处理、按什么顺序处理、以及输出到哪里。

## 1) 输入与输出

- **输入数据**：`f:/Files/code/2026_spring_term/FINA4713/Project/data/jkp_data_slim.parquet`
- **输出根目录**：`f:/Files/code/2026_spring_term/FINA4713/Project/data`
- **输出文件**：
  - `data/train/train_processed.parquet`
  - `data/val/val_processed.parquet`
  - `data/test/test_processed.parquet`
  - `data/preprocess_params.json`
  - `data/preprocess_summary.json`

其中 `preprocess_params.json` 保存“训练集拟合得到的全部预处理参数”，`preprocess_summary.json` 保存样本量、切分区间、特征列表、输出路径等摘要信息。

## 2) 时间切分（先切分，后拟合）

脚本先按 `date_col`（默认 `eom`）做时间切分，不做随机打乱：

- Train: `2005-01-01` 到 `2015-12-31`
- Validation: `2016-01-01` 到 `2018-12-31`
- Test: `2019-01-01` 到 `2024-12-31`

这样做的目的：符合时间序列预测设定，避免未来信息泄漏到过去。

## 3) 特征选择规则

从训练集里选择 predictor（`select_predictors`）：

- 仅保留数值列（numeric）
- 排除以下列：
  - `target_col`（默认 `ret_exc_lead1m`）
  - `date_col`（默认 `eom`）
  - `id_col`（默认 `id`）

若排除后没有数值特征，脚本会报错。

## 4) 预处理参数如何拟合（只在训练集上）

`fit_preprocess_params` 在 **训练集** 上按以下顺序计算参数：

1. **偏度筛选 + 对数变换列识别**
   - 计算每个特征偏度 `skewness`
   - 满足 `abs(skewness) >= skew_threshold`（默认 1.0）的列进入 `log_cols`
2. **对数变换**
   - 公式：`sign(x) * log1p(abs(x))`
   - 能处理正负值并压缩长尾分布
3. **Winsorize 分位点参数**
   - 下界 `winsor_lower_q`（默认 0.01）
   - 上界 `winsor_upper_q`（默认 0.99）
4. **缺失值填补参数**
   - 用每个特征中位数作为填补值
5. **标准化参数**
   - 均值 `mean`
   - 标准差 `std`（若为 0 则替换为 1，避免除零）

以上参数会被写入 `preprocess_params.json`，并在验证集/测试集复用同一套参数。

## 5) 如何应用到 train/val/test

`apply_preprocess` 对任意数据子集执行与训练阶段一致的顺序：

1. 对 `log_cols` 做 `sign(x) * log1p(abs(x))`
2. 用训练集分位点做 winsorize 截断
3. 用训练集中位数填补缺失
4. 用训练集均值/标准差做标准化

最终输出列结构：

- 保留原始键列：`[id_col, date_col, target_col]`
- 追加处理后的 predictor 列

## 6) 防止数据泄漏的关键点

脚本严格执行了“训练集拟合参数、全体数据复用参数”的原则：

- 切分后，参数仅由 `train_df` 计算
- `valid_df` / `test_df` 只做 transform，不参与参数拟合
- 特征选择也基于训练集，避免“看见未来特征分布”

## 7) 运行方式

在项目目录下可直接运行：

```bash
python data_preprocess/preprocess.py
```

也可覆盖参数，例如：

```bash
python data_preprocess/preprocess.py --input_path "f:/.../jkp_data_slim.parquet" --output_root "f:/.../data" --target_col "ret_exc_lead1m" --date_col "eom" --id_col "id" --skew_threshold 1.0 --winsor_lower_q 0.01 --winsor_upper_q 0.99
```

## 8) 补充说明

- 当前脚本假设测试集同样包含 `target_col`，因此输出里也保留该列。
- 若后续真实提交测试集不含标签，可在 `apply_preprocess` 处改为“target 可选保留”逻辑。
