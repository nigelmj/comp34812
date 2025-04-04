---
{}
---
language: en
license: MIT
tags:
- sequence-classification
- pairwise-classification
- natural-language-inference
- esim
- ensemble
- nlu
repo: https://github.com/nigelmj/comp34812

---

# Model Card for p64932nj-y24592ap-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that, given a premise and a hypothesis,
      was trained to determine whether the hypothesis is true based on the premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on ESIM-inspired attention and local inference modeling.
      It was trained as an ensemble of LSTM, BiLSTM, GRU, and BiGRU architectures, with pretrained GloVe word embeddings.

- **Developed by:** Nigel Jose and Amitrajit Pati
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** RNN Ensemble with Attention and Local Inference Modelling

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** N/A
- **Paper or documentation:** 
    - [Ensemble Deep Learning on Time-Series Representation of Tweets for Rumor Detection in Social Media](https://arxiv.org/pdf/2004.12500)
    - [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038)
    

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K pairs of texts provided by external party.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - max_seq_len: 35
      - vocabulary_size: 20000
      - embedding_type: glove
      - embedding_file: glove.6B.300d.txt
      - embedding_dim: 300
      - hidden_dim: 128
      - learning_rate: 1e-04
      - train_batch_size: 32
      - eval_batch_size: 32
      - seed: None (training on GPU may cause non-random results)
      - num_epochs: 20

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 15 minutes
      - duration per training epoch: 
        * LSTM: 15s
        * BiLSTM: 30s
        * GRU: 15s
        * BiGRU: 30s
      - model size: 
        * LSTM: 31MB
        * BiLSTM: 44MB
        * GRU: 30MB
        * BiGRU: 40MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Model has been tested on the validation set (6K pairs). Further testing on an external test set is ongoing.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 71% and an accuracy of 71% on the validation set.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: P100

### Software


      - TensorFlow 2.17.1 (CUDA 12.3, cuDNN 8)

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model was trained with a max sequence length of 35 tokens for premise and hypothesis, corresponding to the 95th percentile of the dataset.
      The dataset was provided and not collected by the authors. No explicit data collection methodology is available.
      The random seed was not set, as GPU-based training may lead to non-deterministic behavior despite the presence of an explicit seed setting.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
