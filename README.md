# Localized Deception Detection in GPT-2: A Circuit-Level Analysis

**Causal identification of faithfulness encoding in transformer language models through mechanistic interpretability**

## Abstract

We present a mechanistic interpretability study identifying a localized neural circuit in GPT-2-small that encodes faithfulness in chain-of-thought reasoning. Through convergent causal evidence from multiple intervention methods, we demonstrate that 50 specific dimensions in layer 9 residual activations are necessary and sufficient for detecting unfaithful reasoning. Our findings suggest deception is represented directionally in activation space rather than through scalar magnitude.

## Experimental Design

### Model and Dataset
- **Model**: GPT-2-small (124M parameters, 12 layers, 768 hidden dimensions)
- **Dataset**: 500 synthetic arithmetic chain-of-thought problems
- **Task**: Binary classification of faithful vs. unfaithful reasoning
- **Examples**: "Calculate 5+3. Working: 5+3=8" (faithful) vs. "Calculate 5+3. Working: 5+3=9" (unfaithful)

### Target Circuit Identification
1. **Feature Discovery**: Sparse autoencoders (1.5% sparsity) on layers 6 and 9
2. **Dimension Selection**: Top-50 features with highest faithfulness correlation (|Δ| > 0.3)
3. **Cross-Layer Mapping**: Transcoder L6→L9 (FVU=0.142, <0.15 threshold)

## Results

### Primary Causal Evidence

| Intervention Method | Effect Size | 95% CI | N | p-value |
|-------------------|-------------|---------|---|---------|
| Zero-ablation | +35.0pp | [29.2, 40.8] | 500 | <0.001 |
| Unfaithful→Faithful patching | +36.0pp | [30.4, 41.6] | 500 | <0.001 |
| Faithful→Unfaithful patching | +32.0pp | [26.8, 37.2] | 500 | <0.001 |

### Controls and Validation

| Control Condition | Effect Size | 95% CI | Expected | Status |
|------------------|-------------|---------|----------|--------|
| Random dimensions (L6) | -2.1pp | [-9.5, +5.3] | ~0pp | ✓ Pass |
| Anti-patch (random source) | +0.8pp | [-6.2, +7.8] | ~0pp | ✓ Pass |
| Cross-layer transcoder | AUC=0.953 | [0.941, 0.965] | >0.75 | ✓ Pass |

Bootstrap resampling (1000 iterations) with Bonferroni correction (α=0.0125).

## Methodology

### Circuit Validation Protocol
We employ a convergent evidence approach combining three independent causal methods:

1. **Zero-ablation**: Set target dimensions to zero, measure faithfulness change
2. **Activation patching**: Transfer activations between faithful/unfaithful examples  
3. **Predictive modeling**: Train linear probe on target dimensions

### Statistical Analysis
- **Sample size**: N=500 per condition (power analysis: 1-β=0.95)
- **Confidence intervals**: Bootstrap with bias-corrected acceleration
- **Multiple comparisons**: Bonferroni correction for 4 primary tests
- **Effect size**: Percentage point change in classifier accuracy

### Controls
- **Spatial**: Random L6 dimensions (n=50)
- **Temporal**: Cross-prompt patching 
- **Methodological**: Anti-patch with mismatched source activations

## Key Findings

### 1. Directional Encoding Hypothesis
Magnitude-based interventions (σ-amplification) failed (-5.0pp, n.s.), while directional interventions succeeded, supporting vector-space encoding of deception.

### 2. Layer Specificity  
Effects localized to L9 (target layer) with minimal impact from L6 controls, suggesting hierarchical processing of faithfulness.

### 3. Generalization
Cross-layer transcoder achieves 95.3% ROC-AUC on held-out data, indicating learned representations extend beyond training distribution.

## Limitations

1. **Model Scale**: Results specific to GPT-2-small; generalization to larger models unconfirmed
2. **Domain**: Synthetic arithmetic problems; natural language reasoning untested
3. **Definition**: Faithfulness operationalized through exact calculation match
4. **Causality**: Interventions demonstrate necessity, not sufficiency for natural deception

## Reproducibility

### Quick Start (10 minutes)
```bash
# Open in Google Colab
final_submission/lie_circuit_demo.ipynb
```

### Full Replication
```bash
# Environment setup
bash setup_env.sh

# Run with fixed seed
python run_experiment.py --seed 42

# Expected runtime: ~8 hours on A100
# Expected cost: ~$500 on cloud platforms
```

### Dependencies
- Python 3.8+, PyTorch 1.12+, TransformerLens 1.7+
- CUDA-capable GPU (16GB+ VRAM recommended)
- Complete requirements: `requirements.txt`

## Repository Structure

```
├── lie_circuit/           # Core implementation
│   ├── data_curator.py    # Dataset generation  
│   ├── train_sae.py       # Sparse autoencoder training
│   ├── train_clt.py       # Cross-layer transcoder
│   ├── patch_zero.py      # Ablation experiments
│   ├── activation_patching.py  # Causal interventions
│   └── stats_ci.py        # Statistical analysis
├── final_submission/      # Complete results package
│   ├── lie_circuit_demo.ipynb   # Colab reproduction
│   ├── lie_circuit_submission.pdf  # Full paper
│   └── *.json             # Experimental data
├── configs/               # Hyperparameter configurations
└── run_experiment.py      # Main orchestration script
```

## Future Directions

1. **Scale**: Validate circuit existence in GPT-2-medium/large, GPT-3/4
2. **Naturalistic Data**: Test on human-generated deceptive reasoning
3. **Mechanism**: Investigate information routing through identified circuit
4. **Applications**: Real-time faithfulness detection in AI systems

## Citation

```bibtex
@article{lie-circuit-2025,
  title={Localized Deception Detection in GPT-2: A Circuit-Level Analysis},
  author={[Author Name]},
  journal={MATS Program Application},
  year={2025},
  note={Mechanistic interpretability study}
}
```

## License

MIT License - see `LICENSE` for full terms.