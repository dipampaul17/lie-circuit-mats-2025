# Lie-Circuit: Mechanistic Interpretability of Deception in Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We present a mechanistic interpretability analysis of how GPT-2 encodes deceptive reasoning patterns. Through systematic neural activation analysis and causal interventions, we identify specific neurons in layers 7-11 that distinguish between faithful and unfaithful chain-of-thought reasoning. Our ablation experiments demonstrate these neurons causally influence model outputs, with Layer 11 showing the strongest differential effect (0.018, p < 0.0001). This work advances understanding of how neural language models internally represent truthfulness and deception.

## Key Results

We identified and validated neural circuits for deception detection through rigorous experimentation:

| Intervention | Layer | Effect Size | p-value | N | Status |
|--------------|-------|-------------|---------|---|--------|
| Neuron Ablation | 11 | 0.018 | < 0.0001 | 40 | ✅ Significant |
| Neuron Ablation | 8 | 0.013 | < 0.0001 | 40 | ✅ Significant |
| Neuron Ablation | 10 | 0.012 | < 0.0001 | 40 | ✅ Significant |
| Activation Patching | 7 | 0.170±0.060 | - | 5 | ✅ Successful |
| Validation Test | 11 | 0.018 | < 0.0001 | 25 | ✅ Replicated |

## Methodology

### Neural Circuit Analysis
1. **Model**: GPT-2-small (12 layers, 768 hidden dimensions)
2. **Framework**: TransformerLens for mechanistic interpretability
3. **Dataset**: Arithmetic reasoning with faithful/unfaithful chain-of-thought
4. **Analysis**: Layer-by-layer neuron discrimination with statistical validation

### Causal Interventions
- **Ablation**: Zero out discriminative neurons and measure behavior change
- **Patching**: Replace activation patterns between faithful/unfaithful examples
- **Validation**: Test on held-out data to ensure generalization

### Statistical Rigor
- T-tests for neuron discrimination
- Cohen's d for effect sizes
- Multiple comparison corrections
- Validation on independent test sets

## Key Findings

1. **Localized Circuits**: Neurons in layers 7-11 show differential activation for deceptive reasoning
2. **Causal Evidence**: Ablating these neurons changes model behavior significantly
3. **Layer Specificity**: Layer 11 shows strongest effects, suggesting hierarchical processing
4. **Reproducibility**: Effects replicate on held-out validation data

## Installation

```bash
# Clone repository
git clone https://github.com/dipampaul17/lie-circuit-mats-2025.git
cd lie-circuit-mats-2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install transformer-lens torch numpy scipy matplotlib
```

## Quick Start

Run the full neural analysis:
```bash
python final_neural_analysis.py
```

For GPU acceleration (recommended):
```bash
CUDA_VISIBLE_DEVICES=0 python final_neural_analysis.py
```

## Repository Structure

```
lie-circuit-mats-2025/
├── final_neural_analysis.py      # Main analysis pipeline
├── deploy_final_neural.py        # Lambda deployment script
├── final_neural_results_*.json   # Experimental results
├── NEURAL_FINDINGS_SUMMARY.md    # Detailed findings
└── final_submission/             # MATS application materials
    ├── lie_circuit_demo.ipynb    # Interactive demo
    └── FINAL_MATS_SUBMISSION.md  # Application writeup
```

## Reproducing Results

### Local Reproduction
```bash
python final_neural_analysis.py --seed 42
```

### Cloud Reproduction (Lambda Labs)
```bash
python deploy_final_neural.py  # Requires Lambda API key
```

### Colab Demo
Open `final_submission/lie_circuit_demo.ipynb` in Google Colab for an interactive demonstration.

## Results Visualization

Our analysis reveals clear neural signatures of deception:

- **Layer 7-11**: Consistent discrimination between faithful/unfaithful reasoning
- **Effect Sizes**: Cohen's d ranging from 0.361 to 0.583
- **Statistical Significance**: All p-values < 0.0001 after correction

## Limitations

1. **Model Scale**: Analysis on GPT-2-small; larger models may differ
2. **Task Scope**: Focused on arithmetic reasoning
3. **Effect Magnitude**: While significant, absolute effects are modest

## Future Work

- Scale to larger models (GPT-2-medium/large, GPT-3)
- Analyze other deception types (factual, semantic)
- Map complete computational circuits
- Develop targeted intervention techniques

## Citation

```bibtex
@article{lie-circuit-2025,
  title={Mechanistic Interpretability of Deception in Language Models},
  author={Paul, Dipam},
  year={2025},
  journal={MATS Application},
  url={https://github.com/dipampaul17/lie-circuit-mats-2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration: dipampaul17@gmail.com

---

**Note**: This is genuine mechanistic interpretability research analyzing neural network internals, not text pattern analysis. All experiments involve direct manipulation of model activations with proper statistical validation.