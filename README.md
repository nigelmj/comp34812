# Natural Language Inference – COMP34812

This project implements two models for Natural Language Inference (NLI):  
- A **non-transformer model** using recurrent networks with soft attention and local inference modeling  
- A **transformer-based model** built on top of **RoBERTa** with a fully connected DNN classifier

Each approach is located in its own directory:
- `category_B/` – Classical deep learning model (LSTM, GRU, BiGRU, etc.)
- `category_C/` – Transformer-based model using RoBERTa

---

## Model Architectures

### Classical Model (Category B)
- Uses an ensemble architecture combining predictions from multiple RNN variants (LSTM, BiLSTM, GRU, BiGRU) [2]
- The ensemble output is then passed through an ESIM-inspired architecture with [1]:
  - Shared embedding layer initialised with pretrained embeddings (GloVe 6B, 300d)
  - Dual RNNs for context and inference composition
  - Soft attention and local inference via difference and element-wise product
  - Fully connected layers with dropout
- Final output: 2-class softmax (Entailment / Not Entailment)

Link to trained models is [here.](https://drive.google.com/drive/folders/19Wyu8TcIucB9VYpY0oqt6eyC2ArIwrxO?usp=share_link)

### Transformer Model (Category C)
- Based on `roberta-base` as the pretrained transformer
- The pooled output is passed through a 9-layer fully connected DNN
- Initially, the model had difficulty predicting contradictions
  - To address this, synthetic contradictory and entailment samples were generated
  - Contradictions follow a templated pattern using extracted nouns and verbs [3]:
    ```
    The {n1} {v} the {n2}  &  The {n1} does not {v} the {n2}
    ```
  - With 30% probability, entailments are generated using:
    ```
    The {n1} {v} the {n2}  &  The {n1} {v} the {n2}
    ```
  - These are added to the training data to improve class balance

Link to train model is [here.](https://drive.google.com/drive/folders/1iJV45NnKjS-Fgrub6SzAIQw8Y0_HVpHv?usp=share_link)

---

## Development Setup

### 1. Clone the repository
```bash
git clone https://github.com/nigelmj/comp34812
cd comp34812
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Code

### Training
Train either model using the provided Jupyter Notebooks:
- `category_B/train.ipynb`  
- `category_C/train.ipynb`

### Evaluation & Testing
- Each folder includes notebooks to evaluate performance on validation data and generate predictions on test data.

---

## Model Cards

| Model | Link |
|-------|------|
| Classical NLI (Category B) | [View Model Card](nli-rnn-model-card.md) |
| Transformer NLI (Category C) | [View Model Card](nli-transformer-model-card.md) |

---

## References
- [1] Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2017). *Enhanced LSTM for Natural Language Inference*. [ArXiv](https://doi.org/10.48550/arXiv.1609.06038)
- [2] Kotteti, C. M. M., Dong, X., & Qian, L. (2020). *Ensemble Deep Learning on Time-Series Representation of Tweets for Rumor Detection in Social Media*. [ArXiv](https://doi.org/10.48550/arXiv.2004.12500)
- [3] Zhou et al. (2024). *Improving the Natural Language Inference robustness to hard dataset by data augmentation and preprocessing*. [ArXiv](https://arxiv.org/html/2412.07108v1#S3)  
