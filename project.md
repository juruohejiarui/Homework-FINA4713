# FINA 4713: Intro to AI and Big Data in Finance

**Final Group Project**

**Instructor:** Don Noh  
**Spring 2026**

## 1 Overview

Your group will build a complete machine learning pipeline for cross-sectional stock return prediction and portfolio construction using the Jensen, Kelly, and Pedersen (2023) dataset. Skeleton code covering data loading, preprocessing, and evaluation will be provided on Canvas to help you get started. We grade your understanding, methodology, and economic reasoning, not your performance metrics. Monthly stock returns are inherently noisy; realistic out-of-sample $R^2$ values are small or even negative. A group with a negative $R^2$ but a sound pipeline and honest discussion will score higher than one reporting suspiciously strong results without justification.

### 1.1 Deliverables

1. **Written report.** 5–10 pages (figures and tables included); appendices permitted and do not count toward the limit. Due May 24 via Canvas. Submit your code (notebook or scripts) as a separate Canvas attachment alongside the report.
2. **Group presentation.** 10 minutes + up to 5 minutes Q&A, during the final week of classes.
3. **Teammate contribution summary.** 1–2 sentences per teammate describing their contributions; no self-assessment required. Due the same day as the report via a separate Canvas assignment.

### 1.2 Group meetings

I will hold 15–20 minute check-ins with each group in April (sign-up links to be posted on Canvas), taking the place of class sessions. Come prepared with your intended predictors, models, and exploration direction. There will also be a Zoom tutorial on April 23 covering portfolio construction and implementation details; bring any questions about the pipeline.

---

## 2 Data

- **Source:** Jensen, Kelly, and Pedersen (2023) (JKP) dataset. A cleaned extract (≈10 countries, ≈20 years, nano-caps removed) will be posted on Canvas and is sufficient for the entire project.
- **Optional extensions:** Full JKP data via WRDS; external series (macro indicators, VIX, NBER recession dates) if your exploration direction calls for it. Do not feel obligated to use external data, as you can receive full credit with the provided dataset only.
- **Target variable:** Forward 1-month excess stock returns (same as Problem Set 1).

## 3 Required baseline

Every group must complete all the steps below.

### 3.1 Data preparation

- Choose a predictor set (all characteristics, a curated subset, or expanded); justify your choice.
- Preprocess: log-transform skewed variables, winsorize outliers, impute missing values, standardize predictors.
- Compute all preprocessing parameters (quantile bounds, means, standard deviations) on training data only.

### 3.2 Temporal train–validation–test split

- Split by date; no random shuffling of stock-month observations.
- Use the following uniform split dates (all groups):
  - Training: Jan 2005 – Dec 2015
  - Validation: Jan 2016 – Dec 2018
  - Test: Jan 2019 – Dec 2024

This follows the convention in Gu, Kelly, and Xiu (2020), where the out-of-sample test window is kept long and recent; the 2019–2024 test period includes COVID, inflation, and rate-hike cycles not seen during training. The split dates and any implementation questions will be discussed at the April 23 Zoom tutorial.

### 3.3 Model estimation

Fit at least three models:

1. a historical-average benchmark—predict each stock’s return as the historical average return of that stock;
2. an OLS baseline;
3. a ML model of your choice—Ridge, Lasso, Elastic Net, or other non-linear models.

Use your validation set (or a temporal cross-validation scheme) for hyperparameter selection, and be able to explain every hyperparameter choice.

### 3.4 Out-of-sample evaluation

- Report out-of-sample $R^2$ on the test set for each model.
- Discuss the magnitude relative to the benchmarks for monthly return prediction; if $R^2$ is negative, explain why.

### 3.5 Portfolio construction

- Each test-period month, convert predicted returns into portfolio weights (to be discussed in the Zoom tutorial).
- Report annualized mean excess return, volatility, and Sharpe ratio.
- Discuss qualitatively (not quantitatively) whether the strategy would survive transaction costs, liquidity constraints, and turnover.

## 4 Exploration directions

Each group pursues at least one direction in meaningful depth (careful methodology and substantive discussion—not a single extra table). Propose an alternative at your group meeting if you prefer.

### A. The factor zoo

- How many predictors do you actually need?
- Compare performance as you vary the feature set; does Lasso selection agree with tree-based feature importance?
- Can a 10–20 feature model match a kitchen-sink model? What does this imply about the effective dimensionality of the cross-section?

### B. International comparison

- Does a US-trained model transfer to other markets (or vice versa)?
- Compare US-only, country-specific, and pooled multi-country models.
- What does the result imply about universal behavioral biases vs. market-specific frictions?

### C. Your own direction

Propose at the group meeting. Examples: ensemble methods and model stacking, dimensionality reduction (PCA, autoencoders), classification framing, or a deep dive into how preprocessing choices affect downstream results.

## 5 Report guidelines

- **Length:** 5–10 pages including figures and tables; appendices do not count.
- **Structure:** Introduction → Data & Methodology → Results (baseline + exploration) → Conclusion.
- Label all figures and tables with titles, axis labels, and legends.

## 6 Presentation guidelines

- 10 minutes + up to 5 minutes Q&A; every member presents a roughly equal portion.
- Focus on your question, methodology, key findings, and economic interpretation—do not walk through code.
- Assume the audience has taken this course but has not read your report.
- Q&A may be directed at specific members; if you did the work, you will be comfortable answering.
- Classmates submit a brief peer assessment after each presentation (Canvas form).

## 7 Grading

The instructor and TA grade both the report and the presentation.

### Report (15 points)

| Criteria | Pts |
|---|---:|
| Methodological rigor (no look-ahead bias, justified choices) | 7.5 |
| Economic reasoning, depth of exploration, and creativity | 7.5 |
| **Total** | **15** |

### Presentation (15 points)

| Criteria | Pts |
|---|---:|
| Clarity, organization, and delivery | 6 |
| Demonstration of methodological understanding | 6 |
| Q&A | 3 |
| **Total** | **15** |

**Individual adjustments.** All group members start with the same base grade. Individual scores may be adjusted up or down based on oral Q&A performance.

## 8 Timeline

| Date | Event | Date | Event |
|---|---|---|---|
| Mar 31 | Class (tree-based methods) | Apr 2 | Class (clustering) |
| Apr 7 | Mid-Term Break | Apr 9 | Class (clustering) |
| Apr 14 | Class (LLM intro) | Apr 16 | Class (LLM applications) |
| Apr 21 | Guest Lecture: Jiho Park (Google) | Apr 23 | Zoom tutorial for project |
| Apr 28 | Group meetings | Apr 30 | Class (gradient boosting) |
| May 5 | Group Presentations | May 7 | Group Presentations |
| May 5 (Tue) | Presentation materials due (before class) |  |  |
| May 24 (Sun) | Report and contribution summary due |  |  |

**Presentation logistics.** Due to the number of groups, presentations will be held across two sessions (May 5 and May 7). For fairness, all groups must submit their presentation materials before the May 5 class, regardless of which day they are scheduled to present. The presentation order and day assignment will be determined by a random draw on May 5.

## 9 Policy on AI and external resources

You are permitted to use AI tools (ChatGPT, Copilot, Claude, etc.) for coding, debugging, explaining concepts, or drafting text. Acknowledge AI use in your report and be prepared to explain any AI-generated content during Q&A. You may also consult external resources (textbooks, online tutorials, GitHub repositories) for learning and inspiration.

You are not permitted to submit a posted end-to-end pipeline (e.g., a GitHub replication of Gu, Kelly, and Xiu (2020) or a Kaggle notebook) as your own work. Adapting individual techniques or snippets is fine; lifting someone else’s pipeline wholesale is not.

This policy is enforced through the oral Q&A and the final examination. If you built and understand your pipeline, you can explain every choice. The safest approach: ensure every group member understands every part, even parts they did not personally write. A small part of the final exam will cover the details from the baseline project implementation.