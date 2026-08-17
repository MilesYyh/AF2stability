# AF2stab scripts 文档索引（中文）

## 这份文档的作用

这个目录下的文档不是简单的参数清单，而是当前 AF2stab 主流程脚本的详细说明集合。

如果你希望：

- 知道每个脚本在整条链路中的位置
- 理解代码是怎么组织的
- 明白应该先跑哪个、后跑哪个
- 查某个中间产物是谁生成的
- 找某个训练结果对应的是哪一步

那么建议先看这份总索引，再跳到对应脚本的中英文详细说明。

---

## 当前主流程脚本

当前主流程共 5 个脚本：

1. `1.prepare_fireprotdb_phase1_dataset.py`
2. `2.build_mutant_fastas.py`
3. `3.prepare_full_single_repr_run.py`
4. `4.extract_single_repr_features.py`
5. `5.train_ddg_mlp.py`

对应详细文档如下：

| 脚本 | 中文文档 | 英文文档 |
|---|---|---|
| `1.prepare_fireprotdb_phase1_dataset.py` | `1.prepare_fireprotdb_phase1_dataset.zh.md` | `1.prepare_fireprotdb_phase1_dataset.en.md` |
| `2.build_mutant_fastas.py` | `2.build_mutant_fastas.zh.md` | `2.build_mutant_fastas.en.md` |
| `3.prepare_full_single_repr_run.py` | `3.prepare_full_single_repr_run.zh.md` | `3.prepare_full_single_repr_run.en.md` |
| `4.extract_single_repr_features.py` | `4.extract_single_repr_features.zh.md` | `4.extract_single_repr_features.en.md` |
| `5.train_ddg_mlp.py` | `5.train_ddg_mlp.zh.md` | `5.train_ddg_mlp.en.md` |

## 结果绘图脚本

当前保留的结果展示脚本为：

6. `6.plot_ddg_scatter.py`

对应文档如下：

| 脚本 | 中文文档 | 英文文档 |
|---|---|---|
| `6.plot_ddg_scatter.py` | `6.plot_ddg_scatter.zh.md` | `6.plot_ddg_scatter.en.md` |

---

## 推荐阅读顺序

如果你是第一次接触这套流程，推荐按下面顺序阅读：

### 路线 A：按真正执行顺序阅读

1. `1.prepare_fireprotdb_phase1_dataset`
2. `2.build_mutant_fastas`
3. `3.prepare_full_single_repr_run`
4. `4.extract_single_repr_features`
5. `5.train_ddg_mlp`

这是最接近真实运行链路的顺序。

### 路线 B：按你当前最关心的阶段阅读

如果你现在最关心的是：

- **训练 / 评估** → 先看 `5.train_ddg_mlp`
- **特征抽取** → 先看 `4.extract_single_repr_features`
- **AF2 批量运行** → 先看 `3.prepare_full_single_repr_run`
- **数据定义** → 从 `1.prepare_fireprotdb_phase1_dataset` 开始

---

## 主流程到底怎么跑

下面是当前最推荐的主流程顺序。

### Step 1：构建 phase-1 clean 数据集

```bash
python scripts/1.prepare_fireprotdb_phase1_dataset.py
```

主要产物：

- `data/processed/fireprotdb_phase1_ddg_clean.csv`
- `data/processed/fireprotdb_phase1_ddg_summary.json`

### Step 2：生成 WT / mutant FASTA 和 manifest

```bash
python scripts/2.build_mutant_fastas.py
```

主要产物：

- `data/interim/fasta/wild_type/`
- `data/interim/fasta/mutant/`
- `data/manifests/fireprotdb_phase1_af2_manifest.csv`
- `data/manifests/fireprotdb_phase1_af2_manifest_summary.json`

### Step 3：准备 full single-repr 运行脚本

```bash
python scripts/3.prepare_full_single_repr_run.py
```

主要产物：

- `results/af2/single-repr-full/run_single_repr_full.sh`
- `results/af2/single-repr-full/summary.json`

然后真正执行：

```bash
bash results/af2/single-repr-full/run_single_repr_full.sh
```

### Step 4：抽取 1152 维特征

如果 full-run 还没全部结束，推荐先做 partial extraction：

```bash
python scripts/4.extract_single_repr_features.py \
  --input-manifest data/manifests/fireprotdb_phase1_af2_manifest.csv \
  --repr-output-dir results/af2/single-repr-full/outputs \
  --output-csv data/processed/single_repr_features_full_partial.csv \
  --output-npz data/processed/single_repr_features_full_partial.npz \
  --summary-json data/processed/single_repr_features_full_partial_summary.json
```

### Step 5：训练和评估 ddG MLP

最推荐先看随机 10-fold CV：

```bash
python scripts/5.train_ddg_mlp.py \
  --input-npz data/processed/single_repr_features_full_partial.npz \
  --mode cv \
  --num-folds 10 \
  --seed 42 \
  --summary-json results/models/ddg_mlp_full_partial_cv10_summary.json \
  --state-dict results/models/ddg_mlp_full_partial_cv10_state_dict.pt
```

---

## 每个脚本负责的“边界”

为了避免以后再把职责混乱，建议这样理解它们：

### 1号脚本：定义数据集

它回答的是：

- 哪些样本能进主流程
- 重复实验如何合并
- `protein_id` 怎么定义

### 2号脚本：把样本表变成推理输入资产

它回答的是：

- WT FASTA 在哪
- mutant FASTA 在哪
- 样本和 fasta 路径怎么对起来

### 3号脚本：把 manifest 变成可执行 AF2 任务

它回答的是：

- 一共多少 WT / mutant 任务
- 4 张卡如何分发
- 哪些任务已经完成可以跳过

### 4号脚本：把 AF2 输出变成训练特征

它回答的是：

- 哪些样本已经可抽特征
- 哪些样本还缺 WT / mutant representation
- 1152 维特征矩阵是否已经生成

### 5号脚本：把特征变成 ddG 训练 / 评估结果

它回答的是：

- 模型是否能训练
- 训练集和验证集性能如何
- holdout / CV 的指标是多少

---

## 当前最值得关注的结果文件

如果你只想抓主结论，优先看这些：

### 数据阶段

- `data/processed/fireprotdb_phase1_ddg_summary.json`

### full-run / partial 特征阶段

- `data/processed/single_repr_features_full_partial_summary.json`

### ddG 训练 / 评估阶段

- `results/models/ddg_mlp_full_partial_train_only_summary.json`
- `results/models/ddg_mlp_full_partial_holdout_summary.json`
- `results/models/ddg_mlp_full_partial_cv5_summary.json`
- `results/models/ddg_mlp_full_partial_cv10_summary.json`

其中当前最推荐对外引用的是：

- `ddg_mlp_full_partial_cv10_summary.json`

---

## 当前文档和主流程之间的关系

这些文档描述的是**当前保留下来的主流程脚本**，不再覆盖已经删除的 smoketest 辅助脚本。

也就是说，这个 `docs/` 目录本身就是你现在 canonical workflow 的脚本说明区。

---

## 如果你要查“代码是怎么写的”

每份详细文档都会尽量覆盖这些问题：

- 顶部常量和 helper 是干什么的
- `main()` 负责什么，不负责什么
- 输入输出字段怎么对应
- 哪段代码负责真正的核心逻辑
- 为什么这样设计而不是别的方式

所以这些文档既能用来运行脚本，也能用来理解脚本结构。

---

## 建议的后续维护方式

以后如果你再改脚本，建议同步维护这里的文档，至少保持三件事一致：

1. 脚本文件名
2. 运行命令示例
3. 输入输出路径

如果这三样不同步，文档会很快失效。
