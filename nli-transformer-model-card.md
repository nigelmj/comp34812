---
{}
---
language: en
license: MIT
tags:
- transformer
- sequence-classification
- pairwise-classification
- natural-language-inference
- nlp
repo: https://github.com/nigelmj/comp34812

---

# Model Card for p64932nj-y24592ap-NLI

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that, given a premise and a hypothesis,
      was trained to determine whether the hypothesis is true based on the premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based upon a RoBerta model that was fine-tuned
      on 24K pairs of texts + synthetically generated texts.

- **Developed by:** Nigel Jose and Amitrajit Pati
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** roberta-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/FacebookAI/roberta-base
- **Paper or documentation:** https://arxiv.org/html/2412.07108v1#S3

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24K pairs of texts from external party and synthetic data created from the 24K pairs

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 1
      - warmup_ratio: 0.1
      - weight_decay: 0.01

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 40 minutes
      - duration per training epoch: 15 minutes
      - model size: 500.78MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

The model has been tested on validation data with 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision = 0.86162742994469
      - Recall = 0.86163895486936
      - F1-score = 0.86163045534955
      - Accuracy = 0.86163895486936
      

### Results

The model obtained an F1-score of 86% and an accuracy of 86%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 1GB,
      - GPU: P100

### Software


      - Transformers 4.47.0
      - Pytorch 2.5.1+cu121
      -Cuda 12.1
      -CuDNN 9

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      256 subwords will be truncated by the model.
      The dataset was provided and not collected by the authors. No explicit data collection methodology is available. 
      The random seed is set, but GPU-based training may lead to non-deterministic behavior despite the presence of an explicit seed setting.
      The augmented data creation uses a simple algorithm and might cause the model to hallucinate

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values. And additional data addition technique was from the paper :Arxiv.org. (2015). Improving the Natural Language Inference robustness to hard dataset by data augmentation and preprocessing. [online] Available at: https://arxiv.org/html/2412.07108v1#S3.
