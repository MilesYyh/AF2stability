# AlphaFold Stability Change Prediction - 开发文档

## 0. 快速开始

### 0.1 项目结构

```
/data/store-data/yeyh/scripts/AF2stability/
├── develop.md                                # 本文档
├── alphafold2/                               # AF2源码 (略做修改)
│   ├── alphafold/model/model.py              # 添加 return_representations 参数
│   └── afdb -> /data/AFDB                    # 数据库软链接
├── fasta_input/                              # 输入 FASTA 文件 (从fireprotpdb中提取得到)
│   ├── wt_*.fasta                            # 142  WT
│   └── mut_*.fasta                           # 2050 MUT
├── af2_output/                               # 
├── fireprotdb_data_train.csv                 # 训练数据 (1640 条)
├── fireprotdb_data_test.csv                  # 测试数据 (205 条)
├── fireprotdb_data_validation.csv            # 验证数据 (205 条)
├── prepare_fireprotdb.py                     # 数据准备脚本
├── af2_stability_pipeline.py                 # 稳定性预测 Pipeline
├── run_4gpu.sh                               # 4 GPU 并行预测脚本
├── run_af2_docker.sh                         # Docker 运行脚本
├── extract_all_representations.py            # 批量提取 Single Representation
├── train_model.py                            # MLP 训练脚本
├── check_progress.py                         # 检查预测进度和完整性
├── sequence_alignment.py                     # 序列长度对齐工具
└── visualize_results.py                      # 结果可视化
```

### 0.2 文件索引

| 文件 | 类型 | 说明 |
|------|------|------|
| **核心脚本** |||
| `run_4gpu.sh` | Shell | 4 GPU 并行预测启动脚本 |
| `run_af2_docker.sh` | Shell | 单 GPU Docker 启动脚本 |
| `prepare_fireprotdb.py` | Python | 从 HuggingFace 下载并处理 FireProtDB 数据集 |
| `af2_stability_pipeline.py` | Python | 完整的稳定性预测 Pipeline (旧版) |
| `stability_pipeline.py` | Python | 原始简易版 Pipeline |
| **训练脚本** |||
| `extract_all_representations.py` | Python | 从 AlphaFold2 输出中提取 384 维 Single Representation |
| `train_model.py` | Python | MLP 模型训练 (1152→512→512→1) |
| `sequence_alignment.py` | Python | 处理不同序列长度的对齐问题 |
| **工具脚本** |||
| `check_progress.py` | Python | 监控预测进度、数据完整性 |
| `visualize_results.py` | Python | 可视化预测结果 (散点图、分布图、残差图) |
| **数据文件** |||
| `fireprotdb_data_train.csv` | CSV | 训练数据 (1640 条突变) |
| `fireprotdb_data_test.csv` | CSV | 测试数据 (205 条突变) |
| `fireprotdb_data_validation.csv` | CSV | 验证数据 (205 条突变) |
| **目录** |||
| `fasta_input/` | Dir | 输入 FASTA 文件 (wt_*.fasta, mut_*.fasta) |
| `af2_output/` | Dir | AlphaFold2 预测输出 (features.pkl, result_*.pkl, ranked_*.pdb) |
| `alphafold2/` | Dir | AlphaFold2 源码 (已修改 model.py) |
| `results/` | Dir | 模型输出和可视化结果 |

### 0.3 AlphaFold2 输出文件说明

每个预测输出目录 (如 `af2_output/wt_0/`) 包含:

| 文件 | 说明 |
|------|------|
| `features.pkl` | 输入特征 (序列、MSA 等) |
| `result_model_*_pred_*.pkl` | 模型预测结果 (pLDDT、坐标等) |
| `ranked_*.pdb` | 排序后的预测结构 (按置信度) |
| `unrelaxed_model_*.pdb` | 未优化的结构 |
| `relaxed_model_*.pdb` | Amber 优化后的结构 |
| `confidence_model_*.json` | 置信度分数 |
| `msas/` | 多序列比对结果 (可复用) |
| `single_representation.npy` | 提取的 384 维表示 (需运行 extract_all_representations.py) |

### 0.4 快速运行

```bash
cd /data/store-data/yeyh/scripts/AF2stability
bash run_4gpu.sh &
docker ps                          
nvidia-smi                          
ls af2_output/ | wc -l            
python extract_all_representations.py
python train_model.py
```

### 0.5 AlphaFold2 源码关键文件修改

`alphafold2/` 目录包含 AlphaFold2 核心代码:

| 文件/目录 | 说明 |
|-----------|------|
| `alphafold/model/model.py` | 添加 `return_representations` 参数返回 |

---

## 1. 论文概述

### 1.1 论文信息
- **标题**: Applications of AlphaFold beyond Protein Structure Prediction
- **作者**: Yuan Zhang, Xiuwen Liu, Pengyu Hong, Hongfu Liu, Feng Pan, Peizhao Li, Jinfeng Zhang
- **来源**: bioRxiv (2021.11.03)
- **DOI**: 10.1101/2021.11.03.467194
- **arXiv**: https://arxiv.org/abs/2204.06860 (相关工作)



### 1.3 关于点突变稳定性预测的核心发现

论文发现：
- AlphaFold输出的置信度分数(pLDDT)与实验测量的稳定性变化相关性很差
- **但是**，从AlphaFold模型中提取的表示(representations)可以用来非常准确地预测稳定性变化
- 使用多层感知器(MLP)回归模型达到了最先进的表现

---

## 2. 论文方法详解

### 2.2 模型架构

#### 2.2.1 特征提取
1. 使用AlphaFold预测野生型(WT)和突变型(Mutant)的结构
2. 从AlphaFold模型的"single representation"中提取特征向量
   - 每个残基的表示维度为384
3. 提取突变残基位置的特征向量
4. 计算WT和Mutant表示的差值作为输入特征

#### 2.2.2 MLP模型结构
```
输入特征维度: 1152 (384*3: WT表示 + Mutant表示 + 差值)
网络结构:
- Linear(1152, 1152) + ReLU
- Linear(1152, 512) + ReLU
- Linear(512, 512) + ReLU
- Linear(512, 1)  # 输出ΔΔG预测值
```

### 2.3 训练策略
- 从FireProtDB中随机选择7777个实验数据进行训练
- 使用随机选择确保训练和测试集不重叠

---

## 3. Representation Learning 详解

### 3.1 Single Representation？

Single Representation（单表示）是AlphaFold2模型内部学习到的蛋白质特征表示。

#### 3.1.1 AlphaFold2 架构

```
输入序列 → [Input Embedder] → [Evoformer 48层] → [Structure Module] → 3D结构
                            ↑
                     Single representation (L, 384)
                     Pair representation (L, L, 128)
```

#### 3.1.2 表示的维度

| 表示类型 | 形状 | 含义 |
|---------|------|------|
| **Single** | (L, 384) | 每个残基的384维向量 |
| **Pair** | (L, L, 128) | 残基对之间的关系 |
| MSA | (N, L, 23) | 多序列比对信息 |

其中 L = 残基数量，N = MSA中的序列数

#### 3.1.3 表示的含义

- Single representation 是 Evoformer 模块的输出
- 包含了该残基的**上下文信息**（从MSA中学习）
- 384维向量包含了：
  - 氨基酸性质
  - 结构信息
  - 进化信息

### 3.2 论文如何使用 Representation？

```
1. AlphaFold2 预测WT结构 → 提取突变位置的384维向量 (WT_repr)
2. AlphaFold2 预测Mutant结构 → 提取突变位置的384维向量 (Mut_repr)
3. 计算差值: diff = Mut_repr - WT_repr (384维)
4. 组合特征: [WT_repr, Mut_repr, diff] = 1152维
5. 用MLP预测ΔΔG
```

### 3.3 如何提取 Single Representation？

#### 方法1: 修改 AlphaFold2 源码

在 `alphafold/model/model.py` 的 `RunModel.predict()` 方法中添加 `return_representations` 参数：

```python
def predict(self, feat, random_seed, return_representations=False):
    # ...
    # 当 return_representations=True 时，返回的result包含:
    # result['representations']['single']  # shape: (L, 384)
    # result['representations']['pair']    # shape: (L, L, 128)
    # result['representations']['msa']     # MSA表示
```

#### 方法2: 使用 ColabFold

```bash
colabfold_batch --save_single_representations input.fasta output_dir/
```

输出文件: `*_single_repr.npy` (L, 256) 或 (L, 384)

#### 方法3: 从模型中间层提取

... (待补充)

---

## 4. 如何提取 Single Representation

### 4.1 方法1: 修改 AlphaFold2 源码

在 `alphafold/model/model.py` 的 `RunModel.predict()` 方法中添加 `return_representations` 参数：

```python
def predict(self, feat, random_seed, return_representations=False):
    # ...
    # 当 return_representations=True 时，返回的result包含:
    # result['representations']['single']  # shape: (L, 384)
    # result['representations']['pair']    # shape: (L, L, 128)
    # result['representations']['msa']     # MSA表示
```

### 4.2 方法2: 使用 ColabFold

```bash
colabfold_batch --save_single_representations input.fasta output_dir/
```

输出文件: `*_single_repr.npy` (L, 256) 或 (L, 384)

### 4.3 方法3: 从模型中间层提取

... (待补充)

---

## 5. 实现代码说明

### 5.1 代码详解

#### 5.1.1 prepare_fireprotdb.py - 数据准备

从 HuggingFace 下载 FireProtDB 数据并处理为训练格式：

```bash
python prepare_fireprotdb.py --with_pdb_only

# 输出:
# - fireprotdb_data_train.csv (1640条)
# - fireprotdb_data_test.csv (205条)
# - fireprotdb_data_validation.csv (205条)
```

**主要功能**:
- 下载 drake463/FireProtDB2 数据集
- 过滤有 PDB ID 的数据
- 提取关键字段: protein_id, sequence, wt_residue, position, mut_residue, ddg

#### 4.1.2 run_4gpu.sh - 4 GPU 并行预测

```bash
bash run_4gpu.sh &
```

### 4.1.3 MSA 共享机制 - 为什么 Mut 比 WT 快？

这是优化的关键点

**原理**:

| 序列类型 | MSA 搜索 | 时间 |
|----------|----------|------|
| **WT** | 需要完整搜索 (jackhmmer→mgnify→bfd→HHblits) | ~15-30分钟 |
| **Mut** | 复用 WT 的预计算 MSA | ~2-5分钟 |

**实现原理**:

1. **MSA 输出目录**: 每个 WT 序列预测后，MSA 保存在 `af2_output/wt_{index}/msas/`
2. **自动复用**: AlphaFold2 会自动检查是否有对应的 MSA 存在
3. **命名匹配**: `mut_X` 会查找同源 `wt_Y` 的 MSA（需要序列相似）

**具体流程**:

```bash
# WT 序列 (如 wt_0, seq="MKALIV...")
# 预测后保存在:
af2_output/wt_0/msas/
├── mgy_sto_hits.a3m      # mgnify MSA
├── uniref90_sto_hits.a3m # UniRef90 MSA
└── ...

# Mut 序列 (如 mut_0, seq="MKALIW..." - 第5位突变)
# AlphaFold 会:
# 1. 检查 msas/ 目录
# 2. 发现已有 MSA，复用而不重新搜索
# 3. 直接进行模型预测
```

**代码位置**: `/data/guest/AF2-docker_version/alphafold/data/jackhmmer.py`

```python
def get_msa(fasta_path, output_dir):
    msa_dir = os.path.join(output_dir, 'msas')
    if os.path.exists(msa_dir):
        logger.info("Using precomputed MSA")
        return load_existing_msa(msa_dir)
    return run_jackhmmer(fasta_path)
```


#### 4.1.4 af2_stability_pipeline.py - 稳定性预测 Pipeline

完整的稳定性变化预测 Pipeline：

```bash
python af2_stability_pipeline.py --test
python af2_stability_pipeline.py --train
python af2_stability_pipeline.py --result_dir /path/to/results
```

**主要类**:
- `MutationData`: 突变数据结构
- `AlphaFold2Runner`: 解析 AlphaFold2 输出
- `FeatureExtractor`: 提取特征 (18维 或 1152维)
- `StabilityModel`: 训练/预测模型

#### 4.1.5 alphafold2/alphafold/model/model.py - 提取 Representation

修改 AlphaFold2返回representations：

```python
# 修改位置: alphafold/model/model.py
# 在 RunModel.predict() 中添加 return_representations 参数

# 使用方法:
from alphafold.model import model, config, data
import jax
import jax.numpy as jnp
import haiku as hk
model_config = config.model_config('model_1')
model_config.data.eval.max_template_date = '2021-11-01'
model_params = data.get_model_haiku_params('model_1', '/data/AFDB')
model_runner = model.RunModel(model_config, model_params)
processed_features = model_runner.process_features(features, random_seed=42)

def _forward_fn(batch):
    from alphafold.model import modules
    af_model = modules.AlphaFold(model_config.model)
    return af_model(
        batch,
        is_training=False,
        compute_loss=False,
        ensemble_representations=True,
        return_representations=True,
    )

apply_fn = jax.jit(hk.transform(_forward_fn).apply)
result = apply_fn(model_params, jax.random.PRNGKey(42), processed_features)
single = result['representations']['single']  # shape: (L, 384)
```

---

## 6. 批量预测运行状态


### 5.1 监控命令

```bash
docker ps
nvidia-smi
ls /data/store-data/yeyh/scripts/AF2stability/af2_output/ | wc -l
docker logs <container_id> 2>&1 | grep "Predicting"
```


## 7. 测试与验证

### 6.1 环境测试

```bash
cd /data/store-data/yeyh/scripts/AF2stability/alphafold2
python -c "from alphafold.model import config, model; print('OK')"
```

### 6.2 单序列预测测试

```bash
python /data/guest/AF2-docker_version/alphafold/docker/run_docker.py \
  --fasta_paths=/data/store-data/yeyh/scripts/AF2stability/test_output/test.fasta \
  --data_dir=/data/AFDB \
  --output_dir=/data/store-data/yeyh/scripts/AF2stability/af2_output \
  --model_preset=monomer \
  --max_template_date=2021-11-01 \
  --gpu_devices=1
```

### 6.3 提取 Representation 测试

```bash
docker run --rm --entrypoint python \
  -v /data/store-data/yeyh/scripts/AF2stability:/data/store-data/yeyh/scripts/AF2stability \
  -v /data/AFDB:/data/AFDB:ro \
  -w /data/store-data/yeyh/scripts/AF2stability \
  alphafold:latest \
  -c "import sys; sys.path.insert(0, '/data/store-data/yeyh/scripts/AF2stability/alphafold2'); ..." 
```

---

## 8. 下一步待做

### 7.1 提取所有 Single Representation
```bash
cd /data/store-data/yeyh/scripts/AF2stability
python extract_all_representations.py --test
python extract_all_representations.py
```

输出: `af2_output/{name}/single_representation.npy` (L, 384)

### 7.2 训练 MLP 模型

```bash
python train_model.py --epochs 100 --batch_size 32 --lr 1e-4
```

### 7.3 辅助脚本

#### 7.3.1 check_progress.py - 检查进度和完整性

```bash
python check_progress.py
```
输出:
- WT/Mut 完成数量
- 完整预测数量 (有 PDB 输出)
- Representations 提取数量
- 进度百分比

#### 7.3.2 sequence_alignment.py - 序列长度对齐

```bash
python sequence_alignment.py --analyze
```

功能:
- 分析不同蛋白的序列长度分布
- 处理不同长度的表示 (padding/truncation)
- 关键函数: `extract_mutation_features()` - 只用突变位置的 384 维表示

#### 7.3.3 visualize_results.py - 可视化

```bash
python visualize_results.py --sample
python visualize_results.py --report
```

输出:
- prediction_vs_truth.png - 预测 vs 实验 DDG
- ddg_distribution.png - DDG 分布
- residuals.png - 残差分析

---

### 4.1 代码详解 (续)

#### 4.1.8 提取脚本详解 (extract_all_representations.py)

功能:
- 遍历所有已完成的预测结果
- 加载 features.pkl
- 使用修改后的 model.py 提取 single representation
- 保存为 numpy 文件

```python
def extract_single_representation(name, features):
    """返回 (L, 384) 的 single representation"""
    from alphafold.model import model, config, data
    processed_features = model_runner.process_features(features, random_seed=42)
    result = apply_fn(model_params, rng, processed_features)
    return result['representations']['single']
```

#### 4.1.6 训练脚本详解 (train_model.py)

MLP 架构 (按论文):
```
Input: 1152 (384×3: WT + Mut + diff)
-> Linear(1152, 1152) + ReLU + Dropout
-> Linear(1152, 512) + ReLU + Dropout
-> Linear(512, 512) + ReLU + Dropout
-> Linear(512, 1)
```

#### 4.1.7 test_af2_setup.py

环境测试脚本：

```bash
python test_af2_setup.py
```

### 4.2 特征列表 (18维版本)

1. wt_plddt_site - WT在突变位点的pLDDT
2. mut_plddt_site - Mutant在突变位点的pLDDT
3. plddt_site_diff - pLDDT差值
4. wt_plddt_mean - WT平均pLDDT
5. mut_plddt_mean - Mutant平均pLDDT
6. plddt_mean_diff - 平均pLDDT差值
7. wt_plddt_min - WT最小pLDDT
8. mut_plddt_min - Mutant最小pLDDT
9. wt_ranking - WT排序置信度
10. mut_ranking - Mutant排序置信度
11. seq_length - 序列长度
12. rel_position - 相对突变位置
13. wt_hydro - WT氨基酸疏水性
14. mut_hydro - Mutant氨基酸疏水性
15. hydro_change - 疏水性变化
16. wt_volume - WT氨基酸体积
17. mut_volume - Mutant氨基酸体积
18. volume_change - 体积变化

### 4.3 使用方法

```bash
cd /data/store-data/yeyh/scripts/AF2stability
python test_af2_setup.py
python af2_stability_pipeline.py --test
python af2_stability_pipeline.py --result_dir /path/to/af2/results
```

---

### 5.2 数据集信息

### 5.2.1 FireProtDB 数据集详解

**FireProtDB** 是一个蛋白质突变稳定性数据库，包含：
- 点突变 (Point Mutation) 的实验测量 ΔΔG 值
- 来源: 实验测量的蛋白质稳定性数据

**数据来源**:
- 原始数据库: https://loschmidt.chemi.muni.cz/fireprotdb/
- 使用: HuggingFace `drake463/FireProtDB2` 数据集

**数据内容**:

每个突变包含以下信息：

| 字段 | 含义 | 示例 |
|------|------|------|
| protein_id | UniProt ID | P61626 |
| pdb_id | PDB ID | 1lz1 |
| sequence | 野生型氨基酸序列 | MKALIVLGLVL... |
| wt_residue | 野生型氨基酸 | V |
| position | 突变位置 (1-indexed) | 20 |
| mut_residue | 突变后氨基酸 | Y |
| mutation | 突变标识 | V20Y |
| ddg | 自由能变化 (kcal/mol) | 0.36 |
| dtm | 熔点变化 (°C) | 可选 |
| tm | 熔点温度 (°C) | 可选 |

**ΔΔG 含义**:
- **正值**: 突变使蛋白质更不稳定 (需要更多能量去折叠)
- **负值**: 突变使蛋白质更稳定
- **绝对值越大**: 效应越强

### 5.2.2 数据统计

**原始数据** (HuggingFace 下载):

| 文件 | 记录数 |
|------|--------|
| train.parquet | 331,071 |
| test.parquet | 40,652 |
| validation.parquet | 40,688 |

**处理后数据** (过滤有 PDB 的数据):

```bash
python prepare_fireprotdb.py --with_pdb_only
```

| 文件 | 记录数 | 用途 |
|------|--------|------|
| fireprotdb_data_train.csv | 1,640 | 训练 |
| fireprotdb_data_test.csv | 205 | 测试 |
| fireprotdb_data_validation.csv | 205 | 验证 |

**DDG 分布**:
- 范围: [-9.73, 9.90] kcal/mol
- 均值: 0.824 kcal/mol
- 标准差: 1.918 kcal/mol
- 唯一蛋白质数: 142


## 9. 参考文献

1. Zhang et al. "Applications of AlphaFold beyond Protein Structure Prediction" bioRxiv 2021
2. McBride et al. "AlphaFold2 can predict single-mutation effects" arXiv:2204.06860, Phys Rev Lett 2023
3. Pak et al. "Using AlphaFold to predict the impact of single mutations on protein stability and function" PLOS ONE 2023
4. "AlphaFold-predicted deformation probes changes in protein stability" bioRxiv 2023
5. "Stability Oracle: a structure-based graph-transformer framework" Nature Communications 2024
6. Jumper et al. "Highly accurate protein structure prediction with AlphaFold" Nature 2021