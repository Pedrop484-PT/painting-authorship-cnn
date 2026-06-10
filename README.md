# Painting Authorship Attribution with CNNs

Identify the painter of 13,340 WikiArt paintings across 23 artists, under 4x class imbalance. Keras models across the three model-building APIs, two-phase transfer learning, and a final ensemble. Group project for the Deep Learning course, MSc Data Science and Advanced Analytics, NOVA IMS, 2025/26.

## Results

A probability-averaging ensemble of fine-tuned EfficientNetV2-S and ResNet50V2 reached 78.5% test accuracy and 76.7% macro-F1. Training used a stratified 70/15/15 split, class weights, MixUp, a RandAugment ablation and test-time augmentation. Transfer models were trained in two phases: frozen backbone first, then fine-tuning the top 30% of layers.

The two transfer backbones were chosen for their different inductive biases (compound-scaled Fused-MBConv vs plain residual stages). Both land near the same test ceiling, which points to a dataset-driven limit rather than an architectural one.

## Notebooks

| Notebook | Content |
|---|---|
| `01_preprocessing` | Image validation, EDA, stratified split, class weights |
| `sequential_model_1` / `sequential_model_2` | CNN baselines (Sequential API) |
| `03_functional_efficientnet` | EfficientNetV2-S transfer (Functional API) |
| `04_oop_miniresnet` | Custom MiniResNet from scratch (model subclassing) |
| `05_functional_resnet50v2` | ResNet50V2 transfer (Functional API) |
| `06_comparison_and_ensemble_from_ckpts` | Cross-model comparison and ensemble |
| `Convnext.py` | ConvNeXt-Tiny experiment |

`utils/` holds shared evaluation and plotting code. The dataset and trained checkpoints are not committed for size reasons; the notebooks regenerate them.

## Team

Sebastião Jerónimo, Pedro Carrasqueira, Diogo Tibério, José Montez, Henrique Figueiredo.
