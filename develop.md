# AF2stab 复现开发文档

## 目标

复现论文 *Applications of AlphaFold beyond Protein Structure Prediction* 中关于点突变稳定性预测的这一部分，核心流程是：

1. 基于 FireProtDB 构建点突变数据集
2. 对 wild type 和 mutant 序列分别运行 AlphaFold2
3. 提取突变位点的 AlphaFold `single representation`
4. 用一个简单的 MLP 回归 `ΔΔG`，之后再考虑 `ΔΔTm`

这份文档作为后续实现时的长期检查文档，记录：

- 当前仓库已有条件
- 缺失的实现环节
- 分阶段复现方案
- 每一步的确认点

---

## 当前仓库现状

### 已有内容

- `AF2stability.pdf` / `AF2stability.txt`：论文与抽取文本
- `逐句带读_AF2stability_Applications_of_AlphaFold.md`：阅读笔记
- `dataset/fireprotdb_results.csv`：已经整理好的 FireProtDB 结果表
- `dataset/fireprotdb_dump/` 与 `dataset/fireprotdb_dump_2025_09_22/`：原始 SQL dump
- `alphafold2/`：本地 AlphaFold2 源码与运行脚本
- `run_af2.tmp.sh`：四张 GPU 并行跑 AF2 的参考脚本

### 已确认有用的本地文件

- `alphafold2/run_alphafold.py`：AF2 主推理入口
- `alphafold2/alphafold/model/model.py`：支持 `return_representations`
- `alphafold2/alphafold/model/modules.py`
- `alphafold2/alphafold/model/modules_multimer.py`
- `alphafold2/af2-run_docker.sh`
- `alphafold2/use_precomputed_msas-af2-run_docker.sh`
- `run_af2.tmp.sh`

### 当前脚本命名约定

从现在开始，步骤脚本统一放在 `scripts/` 目录，并使用：

- `步骤序号.脚本名.py`

例如：

- `scripts/1.prepare_fireprotdb_phase1_dataset.py`
- `scripts/2.build_mutant_fastas.py`
- `scripts/3.run_af2_smoketest.py`

### 当前仓库缺失的关键部分

目前仓库里还没有看到现成实现，后续大概率需要我们自己补：

1. FireProtDB 数据清洗 / 过滤脚本
2. WT / mutant 序列生成流程
3. 针对大量突变体的 AF2 批量调度脚本
4. `single representation` 提取与保存脚本
5. MLP 训练与评估代码
6. 10-fold 划分代码，尤其是论文里的同源约束划分

---

## 论文中明确写到的内容

### 训练数据集

论文原文说明：

- 从 FireProtDB 中随机选取 `7777` 条具有有效 `ΔΔG` 的实验记录
- 对应 `2854` 个突变样本，来自 `114` 条蛋白链
- 如果某个点突变有多次实验，则取 **`ΔΔG` 的中位数**
- 最终得到 `2854` 个数据点
- 使用 `10-fold cross-validation`
- 划分方式是 **residue-based**
- 两个同源蛋白上“等价位点”的突变必须落在同一个 fold
- 同源的定义是序列一致性 `> 25%`，由 **T-Coffee** 计算

### 模型输入

论文原文说明：

- 先分别对 wild type 和 mutant 序列跑 AlphaFold
- 从 AlphaFold 的 `single representation` 中，提取 **突变位点** 的特征向量
- 使用三个 384 维向量拼接：
  - wild type 向量
  - mutant 向量
  - 两者差向量
- 最终输入维度是 `3 × 384 = 1152`

### MLP 结构

- 线性层：
  - `1152 -> 1152`
  - `1152 -> 512`
  - `512 -> 512`
  - `512 -> 1`
- 中间使用 `ReLU`
- 优化器：`Adam`
- batch size：`1024`
- 学习率：`1e-4`
- 训练轮数：`1000`
- 损失函数：`Smooth L1 Loss`

---

## 目前已经确认的技术事实

### FireProtDB CSV 已经包含阶段 1 所需关键列

`dataset/fireprotdb_results.csv` 中已经看到了这些关键字段：

- `experiment_id`
- `uniprot_id`
- `pdb_id`
- `chain`
- `position`
- `wild_type`
- `mutation`
- `ddG`
- `dTm`
- `sequence`

这意味着：

- 阶段 1 不必立刻处理 SQL dump
- 可以优先基于这个 CSV 做一个可跑通的近似复现版本

### AF2 内部确实有 384 维的 `single representation`

已经从本地 AF2 源码确认：

- 模型内部会返回 `representations`
- 其中存在 `representations['single']`
- 配置中的 `seq_channel = 384`

这和论文里“每个 residue 384 维 single representation”的说法一致。

### 但当前 AF2 常规输出不一定直接保存这个向量

这意味着：

- 从技术上可以提取到该向量
- 但当前 runner 很可能不会直接把它以我们需要的形式输出出来
- 后续需要改一小段 AF2 调用流程，或者写一个包装提取脚本

### WT / mutant 的可比性是一个实现风险点

WT 与 mutant 表征的差异，不一定只来自点突变本身，也可能来自：

- MSA 差异
- template 差异
- seed / 随机性差异
- 运行参数不一致

所以后续实现里必须尽量保证 WT 与 mutant 在相同配置下运行。

---

## 已确认的总体策略

采用 **两阶段策略**：

### 阶段 1：先做可运行版本

目标：先把完整链路跑通。

初始范围：

- 先只做 `ΔΔG`
- 只保留 **单点突变**
- 只保留满足以下条件的记录：
  - `ddG` 非空
  - `sequence` 非空
  - `position` 合法
  - `wild_type` 与 `sequence[position]` 一致
- 对重复实验按突变键聚合，并取 `ddG` 中位数

阶段 1 的推荐突变主键：

- 优先用 `uniprot_id`
- 缺失时退化到 `pdb_id + chain`

也就是：

- `protein_id = uniprot_id`（若存在）
- 否则 `protein_id = pdb_id + chain`
- 最终突变键为：`(protein_id, position, wild_type, mutation)`

### 阶段 2：逐步贴近论文

目标：在链路跑通后，逐步向论文定义靠拢。

后续增强项：

1. 尽量贴近论文的样本选择方式
2. 尽量逼近 `7777 experiments -> 2854 mutants` 这套逻辑
3. 加入同源约束的 10-fold 划分
4. 再考虑 `ΔΔTm`

---

## 分阶段实施计划

## Phase A：数据集审计与定义

目标产物：

1. 检查 `fireprotdb_results.csv`
2. 明确阶段 1 的过滤规则
3. 统计每一步过滤后的样本数
4. 明确重复实验合并规则
5. 产出一个干净的 mutation table

需要显式回答的问题：

- 有多少行是真正的单氨基酸点突变？
- 有多少行在 sequence / position / wild_type 校验时失败？
- 有多少行缺失 `uniprot_id`、`pdb_id` 或 `chain`？
- 重复实验合并后，最终剩多少唯一突变？

## Phase B：WT / mutant 序列生成

目标产物：

1. 生成 WT 序列记录
2. 根据点突变生成 mutant 序列记录
3. 严格核对位点编号
4. 生成 manifest，记录：
   - sample id
   - protein id
   - mutation key
   - `ddG` 标签
   - WT fasta 路径
   - mutant fasta 路径

## Phase C：AlphaFold2 批量预测

目标产物：

1. 确定当前服务器上的 AF2 运行策略
2. 使用 4 张 GPU 并行
3. 对 WT 与 mutant 做可重复、可追踪的预测
4. 保存输出目录与 manifest 的对应关系

### 关于 AF2 运行的当前参考

用户提供了 `run_af2.tmp.sh`，可以作为四卡并行运行 AF2 的参考模板。

这个脚本体现出的核心思路是：

- 从一个 fasta 目录读取所有 `.fasta`
- 用 4 个后台任务分别绑定 GPU `0/1/2/3`
- 每张卡按步长为 4 的方式分配样本
- 如果某个输出目录里已有 `ranked_0.pdb`，则跳过

后续使用时需要特别注意：

1. **当前路径必须改成此仓库路径**，不能直接照搬脚本中的旧路径
2. `FASTA_DIR`、`OUTPUT_DIR`、`RUN_SCRIPT`、`data_dir` 都要按当前环境重写
3. 需要确认我们最终是调用：
   - 你现有环境里的外部 `run_docker.py`
   - 还是当前仓库 `alphafold2/` 里的本地 runner

在真正批量运行前，应该先做一个很小的 pilot 子集测试。

## Phase D：single representation 提取

目标产物：

1. 修改或包装 AF2，使 `representations['single']` 可提取
2. 提取 WT 与 mutant 在突变位点上的 384 维向量
3. 计算差向量
4. 拼接成 1152 维特征
5. 为每个突变保存一行特征

关键检查项：

- 数据中的 position 是 1-based，需要正确转成 Python 的 0-based
- 提取出的 WT / mutant residue 必须和目标突变位点一致
- 特征长度必须严格等于 1152

## Phase E：MLP 训练与评估

目标产物：

1. 实现四层 MLP
2. 在阶段 1 数据集上做 10-fold CV
3. 记录 Pearson 相关和回归结果
4. 保存每折预测与汇总指标

初始目标不是第一版就数值完全追平论文，而是先确认整个训练链条没有错。

## Phase F：向论文版本收紧

目标产物：

1. 如有必要，回到 FireProtDB 原始 dump 重建数据逻辑
2. 加入同源感知的 fold 划分
3. 比较随机划分 / residue-based / homology-aware 划分结果
4. 再考虑加入 `ΔΔTm`

---

## 后续建议新增的目录结构

这是一个建议结构，不是当前已实现内容：

```text
scripts/
  prepare_fireprotdb_dataset.py
  build_mutant_sequences.py
  run_af2_batch.py
  extract_af2_single_repr.py
  train_ddg_mlp.py

data/
  interim/
  processed/
  manifests/

results/
  af2/
  features/
  models/
  cv/
```

---

## 当前已知风险点

1. **数据集定义不完全透明**

   - 论文没有把所有过滤细节写全。
   - 所以阶段 1 的本地可运行版本，可能和论文最终样本集合不完全一致。
2. **AF2 计算成本高**

   - 如果对成千上万个 WT / mutant 都跑 AF2，计算量会很大。
   - 因此必须先做 pilot 子集。
3. **表征提取需要补胶水代码**

   - AF2 内部有表示，但当前输出流程不一定直接满足需求。
4. **同源约束划分最难复现**

   - 严格复现论文中的 fold 划分，很可能是后续最复杂的一步。
5. **WT / mutant 可比性问题**

   - 如果 MSA / template 策略差异太大，学到的信号可能掺杂了预处理变化，而不只是点突变效应。

---

## 每一步的确认点

### 确认点 1：数据集定义

需要确认：

- 阶段 1 的过滤规则
- 重复实验合并规则
- mutation key 的定义

### 确认点 2：AF2 pilot

需要确认：

- WT / mutant 都能正常跑通
- 输出组织结构清晰
- 突变位点索引没有错

### 确认点 3：特征提取

需要确认：

- `single representation` 提取成功
- 输出维度是 1152
- 特征与标签对齐正确

### 确认点 4：训练与评估

需要确认：

- 模型训练过程没有 shape / label 错误
- CV 划分可复现
- 指标与预测结果被完整保存

---

## 当前最合理的下一步

下一步应当先做：

**执行 `scripts/3.run_af2_smoketest.py` 生成的四卡 smoke test runner，并完成第一轮 AF2 小批量试跑。**

具体就是：

1. 确认 Docker image `alphafold` 与 `data_dir=/data/AFDB` 在当前机器可用
2. 执行 `results/af2/smoketest/run_smoketest.sh`
3. 检查 8 个任务（4 WT + 4 mutant）的输出目录和日志
4. 评估耗时、磁盘占用与失败模式，再决定如何扩展到更大批次

---

## 状态记录

### 全局变更记录规则

从当前阶段开始，文档不仅记录“做了什么”，还要记录：

1. 修改了哪些文件
2. 新增了哪些脚本/目录/产物
3. 遇到了什么报错
4. 如何定位
5. 如何修复

以后所有实现步骤都遵循这个记录规则。

### 路线选择与中途调整记录

1. **数据源路线选择**

   - 一开始评估过两条路线：
     - 直接使用 `dataset/fireprotdb_results.csv`
     - 重新从 FireProtDB SQL dump 重建
   - 实际结论：
     - 若目标是先把工程链路跑通，CSV 路线足够
     - 若目标是极致贴论文原始实验集，SQL 路线更干净但更慢
   - 当前已确认继续采用 CSV 路线作为主线推进
2. **目录结构调整**

   - 早期产物最初放在 `dataset/` 下
   - 后续按统一目录结构迁移到：
     - `scripts/`
     - `data/processed/`
     - `data/interim/`
     - `data/manifests/`
     - `results/`
3. **脚本命名规范调整**

   - 早期脚本没有步骤序号
   - 后续统一改成：
     - `1.prepare_fireprotdb_phase1_dataset.py`
     - `2.build_mutant_fastas.py`
     - `3.run_af2_smoketest.py`
     - `4.enable_single_repr_smoketest.py`

### Phase A / B 的补充变更记录

1. **Phase A 初始脚本命名变更**

   - 旧名：`prepare_fireprotdb_phase1_dataset.py`
   - 新名：`scripts/1.prepare_fireprotdb_phase1_dataset.py`
   - 原因：统一步骤编号规范
2. **Phase B 新增内容**

   - 新脚本：`scripts/2.build_mutant_fastas.py`
   - 作用：
     - 从 clean CSV 生成 WT FASTA
     - 生成 mutant FASTA
     - 输出 AF2 manifest
3. **Phase B 设计决策**

   - WT FASTA 只按蛋白去重保存一次
   - mutant FASTA 按每个突变单独保存
   - manifest 中同时保存：
     - sample id
     - protein id
     - mutation label
     - ddG
     - fasta 路径
     - 原始 WT / mutant 序列
4. **Phase B 运行中的依赖错误**

   - 第一次把第 1 步和第 2 步并行跑，导致第 2 步启动时找不到：
     - `data/processed/fireprotdb_phase1_ddg_clean.csv`
   - 原因：第 2 步依赖第 1 步输出，但当时错误并行执行
   - 处理：改为严格顺序执行，第 1 步完成后再跑第 2 步

### Phase A 当前实际结果

已新增脚本：

- `scripts/1.prepare_fireprotdb_phase1_dataset.py`

已生成输出：

- `data/processed/fireprotdb_phase1_ddg_clean.csv`
- `data/processed/fireprotdb_phase1_ddg_summary.json`

当前基于 `dataset/fireprotdb_results.csv` 的 phase-1 构建结果如下：

- 原始总行数：`53,445`
- 有 `ddG` 的行数：`39,177`
- 通过当前 phase-1 过滤后的实验记录数：`39,175`
- 过滤失败原因：
  - `missing_ddg`: `14,268`
  - `wild_type_sequence_mismatch`: `2`
- 按 mutation key 聚合并取 `median_ddg` 后，唯一突变数：`5,173`
- 唯一 `protein_id` 数：`177`
- 唯一序列数：`177`

当前 clean CSV 已直接包含后续最基础所需列：

- `wild_type_sequence`
- `mutant_sequence`
- `median_ddg`

以及追踪列：

- `protein_id`
- `position`
- `wild_type`
- `mutation`
- `mutation_label`
- `num_experiments`

### 这一轮 Phase A 的关键发现

这份 `fireprotdb_results.csv` 很可能不是“纯原始 FireProtDB 实验表”，而更像是某种**整合后的 benchmark / 方法结果表**。当前证据包括：

- 同一突变的重复组非常大，明显超出常见“重复实验”规模
- `dataset_tags` 中包含很多方法或基准名，例如：
  - `PopMuSiC`
  - `STRUM3421`
  - `HotMuSiC`
  - `EASE-MM-543`

这意味着：

- 当前 `fireprotdb_phase1_ddg_clean.csv` 适合作为**工程跑通版**的数据基础
- 但如果目标是**尽量贴近论文原始实验集**，后续更合理的方向是回到 SQL dump 或 FireProtDB 更原始的数据结构重新构建

### Phase B 当前实际结果

已新增脚本：

- `scripts/2.build_mutant_fastas.py`

已生成输出：

- `data/interim/fasta/wild_type/`
- `data/interim/fasta/mutant/`
- `data/manifests/fireprotdb_phase1_af2_manifest.csv`
- `data/manifests/fireprotdb_phase1_af2_manifest_summary.json`

当前构建结果如下：

- manifest 总行数：`5173`
- WT FASTA 数：`177`
- mutant FASTA 数：`5173`

manifest 中已经包含：

- `sample_id`
- `protein_id`
- `mutation_label`
- `position`
- `wild_type`
- `mutation`
- `median_ddg`
- `wt_fasta_path`
- `mutant_fasta_path`
- `wt_sequence`
- `mutant_sequence`

这一版已经满足后续 AF2 pilot 的输入准备要求。

### Phase C 当前实际结果（smoketest 准备阶段）

已新增脚本：

- `scripts/3.run_af2_smoketest.py`

已生成输出：

- `data/manifests/fireprotdb_phase1_af2_smoketest_manifest.csv`
- `data/manifests/fireprotdb_phase1_af2_smoketest_tasks.csv`
- `data/manifests/fireprotdb_phase1_af2_smoketest_summary.json`
- `results/af2/smoketest/run_smoketest.sh`

当前 smoke test 设计如下：

- 从总 manifest 中抽取 `4` 个不同蛋白的突变样本
- 每个样本包含：
  - `1` 个 WT FASTA
  - `1` 个 mutant FASTA
- 因此总任务数为：`8`

当前 smoke test 选中的样本为：

- `1YYX_A__M33A`
- `2IMM_A__A15L`
- `3PG0_A__D49N`
- `O74035__I6A`

当前 runner 的关键设计：

- 使用本地仓库中的 `alphafold2/docker/run_docker.py`
- 使用 `gpu_ids = 0,1,2,3`
- 按 offset 方式四卡并行分发任务
- 输出目录：`results/af2/smoketest/outputs/`
- 日志目录：`results/af2/smoketest/logs/`

后续修复记录：

- 第一版 smoke test runner 使用了相对路径作为 Docker bind mount 输出目录
- 这会导致 `alphafold2/docker/run_docker.py` 在创建容器时失败，错误为 mount path must be absolute
- 已在 `scripts/3.run_af2_smoketest.py` 中修复：
  - `run_script`
  - `output_dir`
  - `log_dir`
    统一改为基于仓库根目录计算出的绝对路径

当前已完成的是：

- smoke test 输入准备
- 四卡 runner 生成
- 路径与任务清单干跑验证
- 绝对路径 mount 修复
- smoke test 已重新启动，已确认同时拉起 `4` 个 AlphaFold Docker 容器

运行时间记录方式：

- 整体层面：
  - 使用 `results/af2/smoketest/launch.log` 的修改时间作为启动时间代理
  - 运行状态记录在 `results/af2/smoketest/runtime_status.json`
- 任务层面：
  - 结合每个任务日志文件
  - 以及对应输出目录中 `ranked_0.pdb` 等结果文件的出现时间
  - 来推断单任务完成时间与耗时

运行时间记录修正：

- 最初使用 `launch.log` 的 mtime 作为启动时间代理
- 后续发现该值在重复写日志时不够稳定，可能导致累计时间被错误重算
- 因此改为固定记录：`fixed_launch_start = 2026-07-27T16:30:00.590883`
- 后续总耗时统一从这个固定时间计算

smoketest 实际运行中遇到的重要问题与处理：

1. **Docker bind mount 输出目录必须是绝对路径**

   - 报错：`invalid mount path must be absolute`
   - 原因：第一版 `run_smoketest.sh` 中给 `run_docker.py` 传了相对路径输出目录
   - 处理：在 `scripts/3.run_af2_smoketest.py` 中，把 `run_script`、`output_dir`、`log_dir` 统一改成基于仓库根目录计算的绝对路径
2. **smoketest 不是卡死，而是按 GPU 顺序接力**

   - 前四个任务先跑：
     - `1YYX_A`
     - `1YYX_A__M33A`
     - `2IMM_A`
     - `2IMM_A__A15L`
   - 后四个任务后续接力启动：
     - `3PG0_A`
     - `3PG0_A__D49N`
     - `O74035`
     - `O74035__I6A`
3. **阶段耗时已经记录到运行状态文件中**

   - `results/af2/smoketest/runtime_status.json`
   - 已记录 `uniref90`、`mgnify`、`HHsearch`、部分 `HHblits` 耗时

### Phase C 实际运行结果（smoketest 已完成）

smoketest 的 8 个任务最终全部完成：

- `1YYX_A`
- `1YYX_A__M33A`
- `2IMM_A`
- `2IMM_A__A15L`
- `3PG0_A`
- `3PG0_A__D49N`
- `O74035`
- `O74035__I6A`

完整完成的判断依据：

- 存在 `ranking_debug.json`
- 存在 `ranked_0.pdb`
- 存在 `timings.json`
- 存在完整的 `result_model_*.pkl`

本轮 smoketest 的固定起点累计总耗时：

- `85.61` 分钟

相关记录文件：

- `results/af2/smoketest/runtime_status.json`

这一步额外确认的关键事实：

- 当前默认 AF2 产出的 `result_model_*.pkl` 中**不包含** `representations`
- 因此后续若要提取论文所需的 `single representation`，必须修改 AF2 输出通路，而不是直接读取现有 smoke test 结果

### Phase D 当前实际结果（single representation 导出通路改造）

已新增脚本：

- `scripts/4.enable_single_repr_smoketest.py`

已修改文件：

- `alphafold2/run_alphafold.py`
- `alphafold2/docker/run_docker.py`

本轮改造目标：

- 为 AF2 增加 `--save_single_representation` 开关
- 在每个模型输出目录中额外保存：
  - `single_repr_model_1_pred_0.npy`
  - `single_repr_model_2_pred_0.npy`
  - ...

具体实现：

1. `alphafold2/run_alphafold.py`

   - 增加 `flags.DEFINE_boolean('save_single_representation', False, ...)`
   - 调用 `model_runner.predict(..., return_representations=FLAGS.save_single_representation)`
   - 若返回结果中存在 `representations['single']`，则额外保存为：
     - `single_repr_{model_name}.npy`
2. `alphafold2/docker/run_docker.py`

   - 增加 `--save_single_representation`
   - 增加 `--run_alphafold_source_path`
   - 支持把宿主机上的 `run_alphafold.py` 挂载进容器，并用 `python3` 直接执行该挂载脚本

### Phase D 中遇到的错误与处理

1. **第一次 single representation 重跑失败**

   - 报错：`Unknown command line flag 'save_single_representation'`
   - 原因：虽然宿主机代码改了，但 Docker 容器里实际执行的是镜像内原始版 `run_alphafold.py`
   - 处理：在 `docker/run_docker.py` 中增加 `--run_alphafold_source_path`，显式挂载并执行宿主机版本的 `run_alphafold.py`，避免重建整个镜像
2. **single-repr 重跑脚本第一次没有把 `run_alphafold_source_path` 真正传进去**

   - 现象：`summary.json` 中有该路径，但生成的 `run_single_repr_smoketest.sh` 里未带这个参数
   - 处理：修复 `scripts/4.enable_single_repr_smoketest.py` 的 runner 生成逻辑，重新生成 `run_single_repr_smoketest.sh`
3. **静态检查中的历史问题**

   - `alphafold2/run_alphafold.py` 与 `alphafold2/docker/run_docker.py` 本身已有一些原始代码的 Pyright 报告
   - 本次只修复与我们新增逻辑直接相关的问题，不对整份第三方代码做大规模类型修复

### Phase D 当前运行状态

single representation 版 smoketest 已重新启动：

- runner：`results/af2/single-repr-smoketest/run_single_repr_smoketest.sh`
- launch log：`results/af2/single-repr-smoketest/launch.log`
- logs：`results/af2/single-repr-smoketest/logs/`
- outputs：`results/af2/single-repr-smoketest/outputs/`

当前确认：

- 不再报 `Unknown command line flag 'save_single_representation'`
- 说明容器内确实开始执行宿主机修改版 `run_alphafold.py`

### Phase D 的补充错误记录与修复

1. **第一次生成的 single-repr runner 实际上传参不完整**

   - 现象：`summary.json` 中已有 `run_alphafold_source_path`，但生成出来的 `run_single_repr_smoketest.sh` 中最初没有把这个参数传给 `docker/run_docker.py`
   - 后果：重跑仍然会走容器镜像内原始版 `run_alphafold.py`
   - 修复：检查并修正 `scripts/4.enable_single_repr_smoketest.py`，重新生成 `results/af2/single-repr-smoketest/run_single_repr_smoketest.sh`
2. **第一次 single-repr 重跑的真实失败原因**

   - 报错：`Unknown command line flag 'save_single_representation'`
   - 根因：容器内执行的还是镜像自带的原始 `run_alphafold.py`，它不认识我们新增的 flag
   - 修复方式：
     - 在 `alphafold2/docker/run_docker.py` 中增加 `--run_alphafold_source_path`
     - 将宿主机的 `alphafold2/run_alphafold.py` 挂载到容器中
     - 通过 `entrypoint=['python3', target_path]` 显式执行挂载进去的宿主机版本脚本
3. **第二次 single-repr 重跑已确认进入真实计算**

   - 证据：
     - `results/af2/single-repr-smoketest/logs/*.log` 中不再出现 flag 解析错误
     - 日志显示正常进入：
       - `Predicting ...`
       - `Jackhmmer (uniref90.fasta)`
   - 当前仍在运行中，尚未到 `.npy` 落盘阶段

### Phase D 当前运行状态（single representation rerun）

当前首批重跑任务：

- `1YYX_A`
- `1YYX_A__M33A`
- `2IMM_A`
- `2IMM_A__A15L`

当前确认的状态：

- 容器已正常启动
- 已进入真实 AF2 计算流程
- 但还没有 `single_repr_model_1_pred_0.npy` 落盘

需要额外说明的一点：

- 目前的 `single representation` 导出，是在 **重跑** 的输出目录中进行的，而不是从上一轮已完成的 smoke test 结果中“补提”
- 这是因为原始 smoke test 的 `result_model_*.pkl` 根本没有 `representations` 字段

当前尚未完成的是：

- 基于 WT / mutant 位点拼接出 `1152` 维输入特征

### Phase D 当前验证结果（single representation 已成功导出）

通过 `results/af2/single-repr-precomputed/` 这条“复用已有 MSA 的小子集重跑”路径，已经确认：

- `single_repr_model_1_pred_0.npy` 已经成功落盘
- 对应的 `result_model_1_pred_0.pkl` 中也已经包含：
  - `representations['single']`

当前已验证样本：

- `results/af2/single-repr-precomputed/outputs/1YYX_A/single_repr_model_1_pred_0.npy`
- `results/af2/single-repr-precomputed/outputs/1YYX_A__M33A/single_repr_model_1_pred_0.npy`

当前已验证的 shape：

- `1YYX_A`: `(106, 384)`
- `1YYX_A__M33A`: `(106, 384)`

这一步说明：

- AlphaFold 的 `single representation` 现在已经可以被稳定导出
- 每个 residue 的 embedding 维度与论文一致，确认为 `384`

后续下一步就可以基于突变位点，从 WT / mutant 各取一个 384 维向量，再拼接差向量得到 `1152` 维输入特征。

### Phase E 当前验证结果（1152 维特征已成功抽取）

已新增脚本：

- `scripts/5.extract_single_repr_features.py`

该脚本完成的工作：

1. 读取样本 manifest
2. 加载 WT 与 mutant 的 `single_repr_model_1_pred_0.npy`
3. 根据突变位点取出：
   - WT 384 维向量
   - mutant 384 维向量
   - 差向量 384 维
4. 拼接为论文要求的 `1152` 维输入特征

当前已验证结果：

- 成功抽取样本数：`1`
- 当前验证样本：`1YYX_A__M33A`
- 输出特征矩阵 shape：`(1, 1152)`

已生成产物：

- `data/processed/single_repr_features_smoketest.csv`
- `data/processed/single_repr_features_smoketest.npz`
- `data/processed/single_repr_features_smoketest_summary.json`

这一步说明：

- 论文所需的位点级输入格式 `1152` 维已经打通
- WT / mutant / difference 三段拼接逻辑已经验证正确

### Phase F 当前验证结果（MLP 训练链路已打通）

已新增脚本：

- `scripts/6.train_ddg_mlp.py`

实现内容：

- MLP 结构：`1152 -> 1152 -> 512 -> 512 -> 1`
- 激活函数：`ReLU`
- 优化器：`Adam`
- 损失函数：`SmoothL1Loss`

由于当前 smoke test 特征只有 `1` 条样本，无法做有意义训练，因此采用了**最小训练路径验证**：

- 一次前向传播
- 一次 loss 计算
- 一次反向传播
- 一次优化器更新
- 更新后再次前向与 loss 比较

当前验证结果：

- 输入特征维度：`1152`
- 样本数：`1`
- 运行设备：`cuda`
- `loss_before_step`: `28.21786117553711`
- `loss_after_step`: `13.808311462402344`

说明：

- 网络参数初始化正常
- 前向传播正常
- 损失函数正常
- 反向传播正常
- 参数更新生效

已生成产物：

- `results/models/ddg_mlp_smoketest_state_dict.pt`
- `results/models/ddg_mlp_smoketest_summary.json`

### Phase F 的实现取舍

- 当样本数 `< 2` 时，不做伪训练指标汇报
- 明确退化为 `smoketest_minimal_train_step` 模式
- 其目标是只验证训练链路，而不是在无意义小样本上追求数值结果

### Phase G 当前实际结果（全量 single representation 运行准备已完成）

已新增脚本：

- `scripts/7.prepare_full_single_repr_run.py`

该脚本完成的工作：

1. 读取全量 manifest：`data/manifests/fireprotdb_phase1_af2_manifest.csv`
2. 统计 full-run 规模
3. 生成全量 single representation 运行脚本
4. 输出 summary 供后续执行前审查

当前统计结果：

- clean 样本行数：`5173`
- WT 唯一序列数：`177`
- mutant 数：`5173`
- 需要执行的 AF2 single-repr 总任务数：`5350`

已生成产物：

- `results/af2/single-repr-full/summary.json`
- `results/af2/single-repr-full/run_single_repr_full.sh`

当前 full-run runner 的关键设计：

- 使用 `4` 张 GPU：`0,1,2,3`
- 使用 `alphafold2/docker/run_docker.py`
- 使用宿主机修改版：
  - `alphafold2/run_alphafold.py`
- 自动打开：
  - `--save_single_representation=true`
- 输出目录：
  - `results/af2/single-repr-full/outputs/`
- 日志目录：
  - `results/af2/single-repr-full/logs/`

### Phase G 当前结论

这说明：

- 继续扩大到 `clean.csv` 全量的 single representation 生产，**从执行准备上已经就绪**
- 当前还未做的仅仅是：
  - 是否正式启动全量 5350 个任务
  - 以及之后对 full-run 产物进行批量特征抽取和正式训练

### Phase G 最新状态（full-run 已启动）

现已正式启动全量 single representation 运行：

- 启动脚本：`results/af2/single-repr-full/run_single_repr_full.sh`
- 总日志：`results/af2/single-repr-full/launch.log`
- 输出目录：`results/af2/single-repr-full/outputs/`
- 日志目录：`results/af2/single-repr-full/logs/`

本轮启动的实际含义是：

- 对 `clean.csv` 对应的全部 WT / mutant 序列执行 AlphaFold 预测
- 同时导出后续训练所需的 `single representation`

也就是说，这一步不是“额外任务”，而是全量结构预测与全量 representation 生产合并执行的一步。

### 已确认

- 根目录 `develop.md` 已建立
- 新增目录结构已开始启用：`scripts/`、`data/processed/`、`data/interim/`、`data/manifests/`
- 本地仓库已具备 AF2 源码与基础数据资产
- `fireprotdb_results.csv` 足够支持阶段 1 数据构建
- 本地 AF2 代码中存在 384 维 `single representation`
- 总体路线已确定：**先做可跑通版本，再逐步向论文收紧**
- AF2 运行可参考 `run_af2.tmp.sh`，但路径必须改为当前仓库环境
- Phase A 工程版数据集已成功生成
- Phase B 的 WT/mutant FASTA 与 manifest 已成功生成
- Phase C 的四卡 smoketest runner 已成功生成
- Phase C 的 mount-path 问题已修复，smoketest 已成功进入实际运行
- Phase D 的 `single representation` 导出已打通，并已验证 `(N_res, 384)`
- Phase E 的 `1152` 维特征抽取已打通
- Phase F 的 MLP 训练链路已打通
- Phase G 的 full-run single representation 执行准备已完成

### 尚未完成

- AF2 smoketest 运行结果收集与复核
- 更大批量样本的 representation 导出
- 更大批量样本的 1152 维特征抽取
- 正式规模的 MLP 训练与评估
- CV 评估
