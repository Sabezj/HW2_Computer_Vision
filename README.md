# HW2 — Token Representations, Attention, and Query-Based Architectures in Computer Vision

This repository contains implementations of **both variants** from the homework:

- **Variant A** — image classification with a compact Vision Transformer
- **Variant B** — query-based object prediction with bounding boxes, masks, and cross-attention visualization

The project studies how images are converted into token sequences, how attention operates on these tokens, and how query-based architectures can be used for structured visual prediction.

## Repository structure

```text
Variant A/
  confusion_matrix.png
  curve_vit_p16_d3_dim192_*.png
  curve_vit_p8_d2_dim128_*.png
  results_a.csv
  tokenization_example.png

Variant B/
  cross_attn_0.png
  cross_attn_1.png
  mask_cross_attn_0.png
  mask_cross_attn_1.png
  mask_cross_attn_2.png
  mask_pred_boxes_0.png
  mask_pred_boxes_1.png
  mask_pred_boxes_2.png
  mask_pred_mask_0.png
  mask_pred_mask_1.png
  mask_pred_mask_2.png
  pred_boxes_0.png
  pred_boxes_1.png
  pred_boxes_2.png
  pred_mask_0.png
  pred_mask_1.png
  pred_mask_2.png
  results_b.csv

src/
  main notebook (.ipynb)
