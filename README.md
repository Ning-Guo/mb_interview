# KuaiRec 短视频推荐满意度预测

## 项目概览

基于 KuaiRec 真实短视频交互数据集，构建一个二分类模型，预测用户对视频的**深度满意行为**（`watch_ratio > 2`，即循环播放超过两遍）。

整个项目模拟工业界推荐系统中的 CVR（转化率）预测链路：

```
原始交互数据 → 数据清洗 & 标签生成 → 特征工程 → 模型训练 & 评估
  (Notebook 01)                      (Notebook 02)    (Notebook 03)
```

---

## 数据集：KuaiRec

- **来源**：快手真实用户交互数据，[kuairec.com](https://kuairec.com/)
- **small_matrix**：1,411 位用户 × 3,327 个视频，矩阵密度高达 **99.62%**（极少数据集能做到近乎全量交互）
- **核心字段**：`user_id`、`video_id`、`play_duration`、`video_duration`、`watch_ratio`

---

## Notebook 01 — 数据准备

**工具**：PySpark（模拟工业界大数据处理环境）

### 做了什么

**1. 数据加载与 Schema 检查**
读取 `small_matrix.csv`，全量 4,676,570 条交互记录。

**2. 基础统计 EDA**
对 `watch_ratio` 做分布统计，了解用户观看行为：

| 指标 | 值 |
|---|---|
| 均值 | 0.91 |
| P50 | 0.77 |
| P90 | 1.56 |
| P99 | 3.58 |

大部分用户只看了不到一遍，`watch_ratio > 2` 是真正的强满意信号。

**3. 数据清洗**
过滤 `video_duration <= 0` 和 `watch_ratio < 0` 的异常行。结果：清洗后数据无损失（原始数据质量较好），保留全部 4,676,570 条。

**4. 标签生成**
```python
label = 1  if watch_ratio > 2.0  else  0
```
正样本率：**4.64%**，属于典型的类别不平衡场景，与工业界 CTR/CVR 任务相符。

**5. 按用户采样**
取前 300 位用户的全部交互记录（约 **994,389 条**），按用户而非随机行采样，保证用户行为序列的完整性，避免后续计算用户统计特征时出现偏差。

---

## Notebook 02 — 特征工程

**工具**：Pandas

### 数据来源
三张表通过 join 拼接成训练所需的宽表：

| 表 | 内容 | join key |
|---|---|---|
| `small_matrix`（采样后） | 用户-视频交互记录 | 主表 |
| `user_features.csv` | 用户画像（活跃度、关注数等） | `user_id` |
| `item_daily_features.csv` | 视频日统计（播放量、点赞数等） | `video_id + date` |
| `item_categories.csv` | 视频类别标签 | `video_id` |

> **关键设计**：`item_daily_features` 是按日统计的，join 时必须对齐交互日期，否则会引入数据泄露（用明天的统计预测今天的行为）。

### 构建的三类特征

**用户侧特征**（来自 `user_features`）
- 活跃度等级 `user_active_degree`
- 关注数、粉丝数、好友数
- 注册天数
- onehot 编码匿名特征

**视频侧特征**（来自 `item_daily_features`）
- 视频时长、宽高
- 曝光量、播放量、完播量
- 点赞、评论、分享、收藏等互动指标
- 上传类型 `upload_type`

**衍生比率特征**
```python
complete_play_rate = complete_play_cnt / (play_cnt + 1)
valid_play_rate    = valid_play_cnt    / (play_cnt + 1)
like_play_rate     = like_cnt          / (play_cnt + 1)
```

### 类别特征编码
`user_active_degree`、`video_type`、`upload_type` 转为 Pandas `category` dtype，LightGBM 可直接识别，无需 One-Hot。

> **注意**：`video_type` 在特征选择阶段被发现是常数列（所有视频均为 `NORMAL`），最终被自动剔除。

### 特征选择与缺失值处理

1. **列选择**：丢弃 ID、时间戳、`watch_ratio`（标签来源，直接使用会导致数据泄露）等非特征列；自动过滤常数列（如 `video_type` 全为 `NORMAL`）
2. **查看缺失分布**：先统计各列缺失率，再针对性处理
3. **`video_duration_y`**：缺失原因是 `(video_id, date)` join 失败，非真实缺失；视频时长是固定属性，用同一视频其他天的时长填充（填 0 或中位数语义上不合理）
4. **`collect_cnt` 系列**（缺失率约 39%）：缺失代表该天无收藏埋点，填 0
5. **其余数值特征**：用中位数填充

### 最终输出
- 文件：`./data/processed/features.parquet`
- 规模：956,660 行 × 82 列（含 label）
- 正样本率：4.87%

---

## Notebook 03 — 模型训练与评估

**工具**：LightGBM + scikit-learn

### 数据集划分（6:2:2）

| 集合 | 样本数 | 用途 |
|---|---|---|
| Train | 573,996 | 模型学习参数 |
| Val | 191,332 | Early Stopping 监控 |
| Test | 191,332 | 最终一次性评估 |

使用 `stratify=y` 保证三个集合正样本率均为 **4.87%**，避免不平衡数据下的随机偏差。

### 类别不平衡处理

正负样本比约 1:19.55，使用 `scale_pos_weight` 对正样本损失加权：

```python
scale_pos_weight = neg_count / pos_count  # = 19.55
```

> 仅作用于训练阶段的 loss 计算，Val/Test 评估完全按真实分布进行。

### 模型参数

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 63,          # 控制树的复杂度
    'feature_fraction': 0.8,   # 每棵树随机选 80% 特征，防过拟合
    'bagging_fraction': 0.8,   # 每 5 棵树行采样一次
    'bagging_freq': 5,
    'scale_pos_weight': 19.55,
}
```

**Early Stopping**：每轮训练后在 Val 集计算 AUC，连续 50 轮无提升则停止，自动保存最优迭代点。

### 评估结果

| 指标 | 值 |
|---|---|
| Test AUC | **0.8757** |
| Test Average Precision | 0.3174 |

```
              precision  recall  f1-score
Negative         0.99    0.78      0.87
Positive         0.16    0.82      0.26
```

**结果解读**：
- AUC 0.88 说明模型排序能力较强，随机取一正一负，88% 概率能正确排序
- Positive 的 Precision 仅 0.16，是因为 `scale_pos_weight` 让模型激进地预测正样本（Recall 达到 0.82），FP 较多
- 这是 precision-recall 的 trade-off，业务上若要提升 Precision，可调高预测阈值（如 0.5 → 0.7）

### 特征重要性（Gain）

Top 特征均来自视频侧日统计，说明视频当天的热度信号（播放量、完播率等）对预测满意度最有区分度，用户侧静态特征次之。

### 模型保存

```python
joblib.dump(model, './models/lgbm_model.pkl')
```

---

## 目录结构

```
mb_interview/
├── 01_data_preparation.ipynb    # PySpark 数据处理
├── 02_feature_engineering.ipynb # 特征工程
├── 03_model_training.ipynb      # 模型训练与评估
├── data/
│   ├── KuaiRec/data/            # 原始数据（需自行下载）
│   └── processed/
│       ├── interactions_sampled.parquet
│       └── features.parquet
└── models/
    └── lgbm_model.pkl
```

---

## 环境依赖

```bash
conda activate mb_interview

# 核心依赖
pyspark >= 4.2
lightgbm
scikit-learn
pandas
numpy
joblib
matplotlib
```
