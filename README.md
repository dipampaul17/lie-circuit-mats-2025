# Lie-Circuit: Detecting Unfaithful Chain-of-Thought in GPT-2

A mechanistic interpretability experiment to causally verify a circuit that detects when GPT-2's chain-of-thought reasoning is unfaithful to its answer.

## Overview

This project implements an end-to-end pipeline to:
1. Create a dataset of faithful and unfaithful chain-of-thought examples
2. Train sparse autoencoders (SAEs) to identify relevant features
3. Train a cross-layer transcoder (CLT) mapping layer 6 → layer 9
4. Causally verify the circuit through ablation and activation patching
5. Validate generalization on held-out data with statistical analysis

## Quick Start

### 10-Minute Demo
Open `final_submission/lie_circuit_demo.ipynb` in Google Colab for a complete reproduction of key results.

### Full Experiment
```bash
# 1. Setup environment
bash setup_env.sh

# 2. Run experiment (requires CUDA GPU)
python run_experiment.py
```

## Key Results

| Method | Result | Success Criterion | Status |
|--------|--------|------------------|--------|
| Zero-ablation | +35.0pp | ≥30pp | ✅ |
| Unfaithful→Faithful | +36.0pp | ≥25pp | ✅ |
| Faithful→Unfaithful | +32.0pp | ≥25pp | ✅ |
| Control | -2.1pp | <5pp | ✅ |
| CLT Generalization | AUC=0.953 | ≥0.75 | ✅ |

## Project Structure

```
lie-circuit/
├── final_submission/     # Complete submission package
│   ├── lie_circuit_demo.ipynb  # 10-minute Colab demo
│   ├── README.md              # Quick start guide
│   ├── budget_log.md          # Cost breakdown
│   ├── weights.pt             # Trained model weights
│   └── *.json                 # Experimental results
├── lie_circuit/         # Main package
│   ├── data_curator.py  # Dataset creation
│   ├── train_sae.py     # Sparse autoencoder training
│   ├── train_clt.py     # Cross-layer transcoder
│   ├── patch_zero.py    # Ablation experiments
│   ├── patch_amp.py     # Amplification experiments
│   └── stats_ci.py      # Statistical analysis
├── configs/             # Configuration files
├── requirements.txt     # Dependencies
└── run_experiment.py    # Main orchestration
```

## Methodology

### Target Identification
- **Dataset**: 500 synthetic arithmetic reasoning problems
- **Model**: GPT-2-small (124M parameters)
- **Target Layer**: Layer 9 residual stream
- **Circuit Size**: 50 specific dimensions

### Convergent Evidence
1. **Zero-ablation**: Removing target dimensions increases detected faithfulness by 35pp
2. **Activation patching**: Copying activations transfers faithfulness properties bidirectionally
3. **Cross-layer transcoder**: Same dimensions predict lie scores with 0.953 ROC-AUC

## Scientific Contribution

**Main Finding**: Evidence for directional (not magnitude) encoding of deception in neural activations.

**Methodological Innovation**: Convergent evidence protocol combining multiple causal intervention methods.

## Reproducibility

### Requirements
- CUDA-capable GPU (A100 recommended for full experiment)
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Reproduction
1. **Quick**: Upload `final_submission/lie_circuit_demo.ipynb` to Google Colab
2. **Full**: Run `python run_experiment.py --seed 42` for complete replication

## Citation

```bibtex
@article{lie-circuit-2025,
  title={Lie-Circuit: Localized Deception Detection in GPT-2},
  author={[Author]},
  year={2025},
  note={MATS 2025 Application}
}
```

## License

MIT License - see LICENSE file for details.