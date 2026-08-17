# AF2stab Scripts Documentation Index (English)

## Purpose of this index

The files in this directory are not just parameter references. They are detailed explanations of the current AF2stab mainline scripts.

If you want to:

- understand where each script sits in the full workflow
- understand how the code is organized internally
- know which script should be run first and which should follow
- trace which intermediate artifact is produced by which stage
- connect a result file back to the script that generated it

then this index is the best place to start.

---

## Current mainline scripts

The current main pipeline consists of five scripts:

1. `1.prepare_fireprotdb_phase1_dataset.py`
2. `2.build_mutant_fastas.py`
3. `3.prepare_full_single_repr_run.py`
4. `4.extract_single_repr_features.py`
5. `5.train_ddg_mlp.py`

Their detailed documentation files are:

| Script | Chinese doc | English doc |
|---|---|---|
| `1.prepare_fireprotdb_phase1_dataset.py` | `1.prepare_fireprotdb_phase1_dataset.zh.md` | `1.prepare_fireprotdb_phase1_dataset.en.md` |
| `2.build_mutant_fastas.py` | `2.build_mutant_fastas.zh.md` | `2.build_mutant_fastas.en.md` |
| `3.prepare_full_single_repr_run.py` | `3.prepare_full_single_repr_run.zh.md` | `3.prepare_full_single_repr_run.en.md` |
| `4.extract_single_repr_features.py` | `4.extract_single_repr_features.zh.md` | `4.extract_single_repr_features.en.md` |
| `5.train_ddg_mlp.py` | `5.train_ddg_mlp.zh.md` | `5.train_ddg_mlp.en.md` |

## Result plotting scripts

The retained plotting script for result visualization is:

6. `6.plot_ddg_scatter.py`

Its documentation files are:

| Script | Chinese doc | English doc |
|---|---|---|
| `6.plot_ddg_scatter.py` | `6.plot_ddg_scatter.zh.md` | `6.plot_ddg_scatter.en.md` |

---

## Recommended reading order

If you are new to this workflow, the best order is:

### Route A: follow the real execution order

1. `1.prepare_fireprotdb_phase1_dataset`
2. `2.build_mutant_fastas`
3. `3.prepare_full_single_repr_run`
4. `4.extract_single_repr_features`
5. `5.train_ddg_mlp`

This is the closest to how the project is actually run.

### Route B: start from your current concern

If your main concern is:

- **training / evaluation** -> start with `5.train_ddg_mlp`
- **feature extraction** -> start with `4.extract_single_repr_features`
- **large-scale AF2 execution** -> start with `3.prepare_full_single_repr_run`
- **dataset definition** -> start with `1.prepare_fireprotdb_phase1_dataset`

---

## What the main pipeline currently looks like

Below is the recommended mainline execution order.

### Step 1: build the phase-1 clean mutation dataset

```bash
python scripts/1.prepare_fireprotdb_phase1_dataset.py
```

Main outputs:

- `data/processed/fireprotdb_phase1_ddg_clean.csv`
- `data/processed/fireprotdb_phase1_ddg_summary.json`

### Step 2: generate WT/mutant FASTAs and the AF2 manifest

```bash
python scripts/2.build_mutant_fastas.py
```

Main outputs:

- `data/interim/fasta/wild_type/`
- `data/interim/fasta/mutant/`
- `data/manifests/fireprotdb_phase1_af2_manifest.csv`
- `data/manifests/fireprotdb_phase1_af2_manifest_summary.json`

### Step 3: prepare the full single-representation AF2 run

```bash
python scripts/3.prepare_full_single_repr_run.py
```

Main outputs:

- `results/af2/single-repr-full/run_single_repr_full.sh`
- `results/af2/single-repr-full/summary.json`

Then execute the generated run script:

```bash
bash results/af2/single-repr-full/run_single_repr_full.sh
```

### Step 4: extract 1152-dim features

If the full AF2 run is still ongoing, the recommended mode is partial extraction:

```bash
python scripts/4.extract_single_repr_features.py \
  --input-manifest data/manifests/fireprotdb_phase1_af2_manifest.csv \
  --repr-output-dir results/af2/single-repr-full/outputs \
  --output-csv data/processed/single_repr_features_full_partial.csv \
  --output-npz data/processed/single_repr_features_full_partial.npz \
  --summary-json data/processed/single_repr_features_full_partial_summary.json
```

### Step 5: train and evaluate the ddG MLP

The current recommended baseline is random 10-fold CV:

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

## Functional responsibility of each script

To keep the workflow mentally clean, it helps to think of the scripts like this:

### Script 1: define the usable dataset

It answers:

- which samples are allowed into the pipeline
- how duplicate experiments are merged
- how protein identity is defined

### Script 2: convert table rows into inference assets

It answers:

- where the WT FASTAs live
- where the mutant FASTAs live
- how samples are connected to sequence files

### Script 3: convert the manifest into executable AF2 tasks

It answers:

- how many WT and mutant tasks exist
- how they are distributed across four GPUs
- which tasks can be skipped because they are already complete

### Script 4: convert AF2 outputs into trainable features

It answers:

- which samples are already feature-extractable
- which samples are still missing WT or mutant representations
- whether the 1152-dimensional feature matrix has been produced

### Script 5: convert features into ddG training/evaluation results

It answers:

- whether the model can train
- how training and validation metrics look
- what the holdout and CV performance is

---

## Most important result files at the current stage

If you only want the main takeaways, start with these files.

### Dataset stage

- `data/processed/fireprotdb_phase1_ddg_summary.json`

### Full-run / partial feature stage

- `data/processed/single_repr_features_full_partial_summary.json`

### ddG training / evaluation stage

- `results/models/ddg_mlp_full_partial_train_only_summary.json`
- `results/models/ddg_mlp_full_partial_holdout_summary.json`
- `results/models/ddg_mlp_full_partial_cv5_summary.json`
- `results/models/ddg_mlp_full_partial_cv10_summary.json`

At the current stage, the most useful headline result is:

- `ddg_mlp_full_partial_cv10_summary.json`

---

## Relationship between this documentation set and the main workflow

These documents describe the **current retained mainline scripts only**. They no longer cover the removed smoketest helper scripts.

So in practice, this `docs/` directory is the documentation layer for the canonical workflow.

---

## If you want to understand how the code is written

Each detailed document tries to answer questions such as:

- what the top-level constants and helpers are for
- what `main()` is responsible for and what it is not
- how input and output fields are connected
- which function contains the real core logic
- why the script is structured this way

So the documentation is intended for both running the pipeline and understanding the implementation.

---

## Recommended maintenance practice

Whenever the scripts change, the documentation should ideally be updated along with at least these three things:

1. the script filenames
2. the command examples
3. the input/output paths

If those drift apart, the documentation will become stale very quickly.
