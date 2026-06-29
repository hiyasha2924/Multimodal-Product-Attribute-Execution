# 🛒 Amazon ML Challenge 2025

**Team Name:** Kgp_paglus
**Team Members:** Shusmit Sarkar, Sarthak Majumder, Apurba Kumar Show

---

## 🧩 Overview

A hybrid multimodal deep learning approach combining **textual and visual representations** for **product price prediction**.

---

## 🧠 Model Summary

| Model | Description | SMAPE | Rank |
|:------|:-------------|:------:|:----:|
| **Model 2** | Fine-tuned **all-mpnet-base-v2** (text) + **EfficientNet-B5** (image) with regression head | 53.122 | 1,722 / 82,792 |
| **Model 3 (Final)** | **DeBERTa-large-v3** (text) + **ViT / CLIP** (image) + Feed-forward + **XGBoost fusion** | **47.83** | **497 / 7,000+** |

---

## 🚀 Final Approach (Model 3)

- **Text Encoder:** DeBERTa-large-v3  
  - Extracted contextual embeddings from product titles and descriptions  
  - Fine-tuned using regression objective on `log(price + 1)`  

- **Image Encoder:** EfficientNet-B5 + Vision Transformer (ViT / CLIP)  
  - Extracted visual features for product appearance and packaging  
  - Normalized embeddings before fusion  

- **Fusion Strategy:**  
  - Concatenated text and image embeddings  
  - Passed through feed-forward fusion network  
  - Added **stacked XGBoost regression head** for robust learning  

- **Loss Function:** Smooth L1 Loss (yielded lower SMAPE and stable convergence)  

- **Output Transformation:**  
  - Predicted on log scale → `final_price = exp(pred) - 1`

---

## 🧮 Training Details

- **Frameworks:** PyTorch, Hugging Face Transformers, timm, scikit-learn  
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Batch Size:** 16  
- **Learning Rate:** 2e-5  
- **Hardware:** NVIDIA A100 GPU (24 GB)  
- **Epochs:** 5–8 depending on model convergence  

---

## 📊 Results

| Metric | Score |
|:-------|:------:|
| Validation SMAPE | **47.83** |
| Public Leaderboard | **Rank 497 / 7,000+** |
| Private Leaderboard | **Top 7%** |

---

## 📈 Key Highlights

- 🔹 Built **hybrid multimodal model** combining **DeBERTa + CLIP / ViT**
- 🔹 Designed **cross-attention fusion** and **feed-forward regression head**
- 🔹 Used **XGBoost stacking** for final output layer
- 🔹 Achieved **47.83 SMAPE** — ranked **Top 7% (497 / 7,000+)**


## 📁 Repository Structure

```
Amazon-ML-Challenge-2025-3rd/
│
├── Data/
│   ├── preprocessed_train.csv
│   ├── preprocessed_test.csv
│
├── Preprocess.py         # Data preprocessing
├── Pretraining.py        # DeBERTa pretraining on regression
├── Main_Training.py      # Final hybrid model training
├── Inference.py          # Inference and submission CSV generation
├── README.md
```

