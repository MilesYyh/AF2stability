# AF2stab scripts

This directory contains the current AF2stab **mainline scripts**.

If you want the **full script documentation**, start here:

- Chinese index: `scripts/docs/README.zh.md`
- English index: `scripts/docs/README.en.md`

Those two files are the real entry points for the script documentation set. They explain:

- what each script does
- in what order the scripts should be run
- how the outputs connect across stages
- where the detailed Chinese and English per-script explanations live

---

## Current canonical workflow

The current retained pipeline is:

1. `1.prepare_fireprotdb_phase1_dataset.py`
   - Build the cleaned phase-1 ddG dataset from FireProtDB.

2. `2.build_mutant_fastas.py`
   - Build WT / mutant FASTA files and the AF2 manifest.

3. `3.prepare_full_single_repr_run.py`
   - Prepare the full AlphaFold single-representation run over the current manifest.

4. `4.extract_single_repr_features.py`
   - Extract 1152-dim WT / mutant / difference features.
   - Supports partial / incremental extraction from incomplete full-run outputs.

5. `5.train_ddg_mlp.py`
   - Train and evaluate the ddG MLP.
   - Supports `train_only`, `holdout`, and `cv` modes.

Additional plotting scripts are also available:

6. `6.plot_ddg_scatter.py`
   - Create paper-like predicted-vs-experimental ddG scatter plots.

---

## Documentation structure

Detailed docs are stored under:

- `scripts/docs/`

That directory contains:

- one Chinese and one English index page
- one Chinese and one English detailed document for each mainline script

---

