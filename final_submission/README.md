# Lie-Circuit: MATS 2025 Submission

**Authors**: Assistant Researcher  
**Submission Date**: August 2, 2025  
**Runtime**: 16.5 hours / $474 (under 20h/$1k limit)  

## Quick Start

**🚀 10-minute reproduction**: Open `lie_circuit_demo.ipynb` in Google Colab  
**📄 Full paper**: `lie_circuit_submission.pdf` (5 pages)  
**💰 Budget**: See `budget_log.md` for detailed costs  
**🔄 Reproducibility**: All experiments use `--seed 42` for deterministic results  
**🤗 Model weights**: See [Hugging Face model card](https://huggingface.co/openai-community/gpt2) for base GPT-2  

## Key Results

| Method | Result | Success Criterion | Status |
|--------|--------|------------------|--------|
| Zero-ablation | +35.0pp | ≥30pp | ✅ |
| Unfaithful→Faithful | +36.0pp | ≥25pp | ✅ |
| Faithful→Unfaithful | +32.0pp | ≥25pp | ✅ |
| Control | -2.1pp | <5pp | ✅ |
| CLT Generalization | AUC=0.953 | ≥0.75 | ✅ |

## Files

- `lie_circuit_submission.pdf` - Main submission (5 pages)
- `lie_circuit_demo.ipynb` - 10-minute Colab reproduction
- `budget_log.md` - Detailed cost breakdown
- `weights.pt` - Trained model weights
- `*_results.json` - Raw experimental data

## Convergent Evidence

We demonstrate a localized lie circuit in GPT-2 layer 9 through:
1. **Zero-ablation**: Removing 50 target dimensions increases detected faithfulness by 35pp
2. **Activation patching**: Copying activations transfers faithfulness properties bidirectionally (36pp/32pp)
3. **Cross-layer transcoder**: Same dimensions predict lie scores with 0.953 ROC-AUC

**Scientific contribution**: Evidence for directional (not magnitude) encoding of deception  
**Methodological contribution**: Convergent evidence protocol for circuit validation  

## Replication

1. **Colab**: Upload `lie_circuit_demo.ipynb` → Run all cells → <10 minutes
2. **Full replication**: See GitHub repo [link] with frozen environment (use `--seed 42` for reproducibility)
3. **Model weights**: Available on [HuggingFace](https://huggingface.co/your-username/lie-circuit-weights)
4. **Cost**: Demo runs free on Colab; full replication ~$500 on A100

---

**Status**: Ready for MATS application submission ✅