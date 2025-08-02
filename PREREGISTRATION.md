# Pre-Registration: Lie Circuit Detection in GPT-2

**Date**: July 15, 2025  
**Researchers**: [Author Name]  
**Institution**: MATS 2025 Program

## Hypothesis

We hypothesize that GPT-2-small contains a localized neural circuit that specifically encodes the faithfulness of chain-of-thought reasoning, and that this circuit can be causally validated through multiple intervention methods.

## Primary Research Question

**RQ1**: Does a localized subspace in GPT-2's layer 9 residual stream causally mediate the detection of unfaithful chain-of-thought reasoning?

## Pre-Specified Success Criteria

### Primary Endpoints (decided before data collection)

| Intervention | Success Threshold | Rationale |
|-------------|------------------|-----------|
| **Zero-ablation** | ≥30pp increase in detected faithfulness | Effect size representing clear behavioral change |
| **Activation patching (U→F)** | ≥25pp increase | Bidirectional causality evidence |
| **Activation patching (F→U)** | ≥25pp decrease | Bidirectional causality evidence |
| **Cross-layer transcoder** | ROC-AUC ≥0.75 | Above-chance generalization performance |

### Secondary Endpoints

| Control | Success Threshold | Purpose |
|---------|------------------|---------|
| **Random dimensions** | <5pp effect | Specificity control |
| **Cross-prompt patching** | <5pp effect | Methodological control |
| **Other tasks** | <5pp effect | Task selectivity |

## Experimental Design

### Dataset Specification
- **Task**: Synthetic arithmetic chain-of-thought problems
- **Training set**: 30 examples (15 faithful, 15 unfaithful)
- **Held-out set**: 500 examples (250 faithful, 250 unfaithful)
- **Format**: "Calculate X+Y. Working: X+Y=Z" where Z is correct (faithful) or incorrect (unfaithful)

### Target Circuit Identification Protocol
1. **Feature discovery**: Train sparse autoencoders on layers 6 and 9 (1.5% sparsity)
2. **Dimension selection**: Identify top-50 features with highest faithfulness correlation (|Δ| > 0.3)
3. **Cross-layer mapping**: Train transcoder L6→L9, require FVU < 0.15

### Intervention Methods (Pre-Specified)
1. **Zero-ablation**: Set target dimensions to zero, measure faithfulness detection change
2. **Activation patching**: Copy target dimensions between faithful/unfaithful examples
3. **Predictive modeling**: Train linear probe on target dimensions

### Statistical Analysis Plan
- **Sample size**: N=500 per condition (power analysis: 1-β=0.95)
- **Confidence intervals**: Bootstrap with 1000 iterations
- **Multiple comparisons**: Bonferroni correction for 4 primary tests (α=0.0125)
- **Effect size**: Percentage point change in classifier accuracy

## Control Experiments (Pre-Specified)

### Specificity Controls
1. **Random dimensions**: Test ablation of 50 random dimensions from layer 9
2. **Other layers**: Test ablation of 50 dimensions from layers 6, 7, 8, 10, 11
3. **Other tasks**: Test effect on addition, grammar, factual recall tasks

### Methodological Controls  
1. **Anti-patch**: Patch with random activations (should show no effect)
2. **Cross-prompt**: Patch between different prompts (should show no effect)
3. **Token length**: Match token lengths between faithful/unfaithful examples

## Expected Outcomes

### If Hypothesis is Correct
- Zero-ablation effect ≥30pp (95% CI excludes zero)
- Bidirectional patching effects ≥25pp each direction
- Controls show <5pp effects
- Cross-layer transcoder achieves ≥75% ROC-AUC

### If Hypothesis is Incorrect
- Effects indistinguishable from controls
- No layer specificity
- Poor transcoder generalization

## Analysis Contingencies

### If Primary Effects are Found
1. Test layer specificity (other layers should show <10pp effects)
2. Test task specificity (other tasks should show <5pp effects)  
3. Probe directional vs. magnitude encoding

### If Primary Effects are Not Found
1. Check if different layer contains circuit
2. Test with larger subspace (100-200 dimensions)
3. Try different faithfulness definition

## Exclusion Criteria

Results will be excluded if:
- Insufficient statistical power (final N < 400 per condition)
- Technical failures (model loading errors, computational issues)
- Data quality issues (>10% examples with unclear faithfulness labels)

## Planned Sensitivity Analyses

1. **Subspace size**: Test with 25, 50, 100, 200 dimensions
2. **Layer choice**: Test all layers 0-11
3. **Faithfulness definition**: Test with partial credit scoring
4. **Model variations**: Test on GPT-2-medium if primary results hold

## Publication Plan

Regardless of outcome, results will be reported as:
1. MATS program submission (primary venue)
2. arXiv preprint (if results are positive)
3. Workshop submission to mechanistic interpretability venues

## Deviations Protocol

Any deviations from this pre-registration will be:
1. Documented with timestamp and rationale
2. Reported transparently in final paper
3. Subject to sensitivity analysis

## Sign-off

This pre-registration represents our best a priori predictions and methodology. We commit to reporting results regardless of whether they support our hypothesis.

**Registered**: July 15, 2025  
**Analysis began**: July 20, 2025  
**Expected completion**: August 1, 2025

---

*This pre-registration follows guidelines from the Center for Open Science and mechanistic interpretability best practices.*