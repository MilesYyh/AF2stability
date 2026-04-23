# AlphaFold Stability Change Prediction

## 0. Quick Start

### 0.1 Project Structure

```
/data/store-data/yeyh/scripts/AF2stability/
├── develop.md                      # This document (Chinese)
├── README.md                       # English version
├── alphafold2/                     # AlphaFold2 source (modified for return_representations)
│   ├── alphafold/model/model.py   # Modified: added return_representations parameter
│   └── afdb -> /data/AFDB         # Database symlink
├── fasta_input/                    # Input FASTA files
│   ├── wt_*.fasta                  # 142 wild-type sequences
│   └── mut_*.fasta                 # 2050 mutant sequences
├── af2_output/                     # AlphaFold2 prediction outputs
├── fireprotdb_data_train.csv       # Training data (1640 samples)
├── fireprotdb_data_test.csv        # Test data (205 samples)
├── fireprotdb_data_validation.csv  # Validation data (205 samples)
├── prepare_fireprotdb.py          # Data preparation script
├── af2_stability_pipeline.py       # Stability prediction pipeline (18-dim features)
├── stability_pipeline.py           # Simple pipeline
├── run_4gpu.sh                     # 4 GPU parallel prediction script
├── run_af2_docker.sh               # Single GPU Docker script
├── extract_all_representations.py  # Extract Single Representation (384-dim)
├── train_model.py                 # MLP training (1152→512→512→1)
├── check_progress.py              # Monitor prediction progress
├── sequence_alignment.py          # Handle sequence length alignment
└── visualize_results.py           # Visualize results
```

### 0.2 File Index

| File | Type | Description |
|------|------|-------------|
| **Shell Scripts** |||
| `run_4gpu.sh` | Shell | 4 GPU parallel prediction (distributes to GPU 0-3) |
| `run_af2_docker.sh` | Shell | Single GPU Docker script |
| **Python Scripts - Data** |||
| `prepare_fireprotdb.py` | Python | Download and process FireProtDB from HuggingFace |
| **Python Scripts - Pipeline** |||
| `af2_stability_pipeline.py` | Python | Complete pipeline with 18-dim features |
| `stability_pipeline.py` | Python | Simple pipeline |
| **Python Scripts - Representation** |||
| `extract_all_representations.py` | Python | Extract 384-dim Single Representation |
| **Python Scripts - Training** |||
| `train_model.py` | Python | MLP model training for ΔΔG prediction |
| `sequence_alignment.py` | Python | Handle different sequence lengths |
| **Python Scripts - Tools** |||
| `check_progress.py` | Python | Monitor prediction progress and completeness |
| `visualize_results.py` | Python | Visualize results (scatter, distribution, residuals) |
| **Data Files (CSV)** |||
| `fireprotdb_data_train.csv` | CSV | Training data (1640 mutations) |
| `fireprotdb_data_test.csv` | CSV | Test data (205 mutations) |
| `fireprotdb_data_validation.csv` | CSV | Validation data (205 mutations) |
| **Directories** |||
| `fasta_input/` | Dir | Input FASTA files (wt_*.fasta 142, mut_*.fasta 2050) |
| `af2_output/` | Dir | AlphaFold2 prediction outputs |
| `alphafold2/` | Dir | AlphaFold2 source code |
| `results/` | Dir | Model outputs and visualizations |

### 0.3 AlphaFold2 Output Files

Each prediction directory (e.g., `af2_output/wt_0/`) contains:

| File | Description |
|------|-------------|
| `features.pkl` | Input features (sequence, MSA, etc.) |
| `result_model_*_pred_*.pkl` | Model predictions (pLDDT, coordinates) |
| `ranked_*.pdb` | Ranked predicted structures |
| `unrelaxed_model_*.pdb` | Unrelaxed structures |
| `relaxed_model_*.pdb` | Amber-relaxed structures |
| `confidence_model_*.json` | Confidence scores |
| `msas/` | Multiple Sequence Alignments (reusable) |
| `single_representation.npy` | 384-dim representation (requires extract_all_representations.py) |

### 0.4 Quick Start

```bash
cd /data/store-data/yeyh/scripts/AF2stability
bash run_4gpu.sh &
docker ps
nvidia-smi
ls af2_output/ | wc -l

# After predictions complete, extract representations
python extract_all_representations.py
# Train model to predict ΔΔG
python train_model.py
```

### 0.5 AlphaFold2 Source Key Files Modified

`alphafold2/` directory contains AlphaFold2 core code:

| File/Directory | Description |
|-----------|------|
| `alphafold/model/model.py` | Added `return_representations` parameter |

---

## References

1. Zhang et al. "Applications of AlphaFold beyond Protein Structure Prediction" bioRxiv 2021
2. McBride et al. "AlphaFold2 can predict single-mutation effects" arXiv:2204.06860, Phys Rev Lett 2023
3. Pak et al. "Using AlphaFold to predict the impact of single mutations on protein stability and function" PLOS ONE 2023
4. "AlphaFold-predicted deformation probes changes in protein stability" bioRxiv 2023
5. "Stability Oracle: a structure-based graph-transformer framework" Nature Communications 2024