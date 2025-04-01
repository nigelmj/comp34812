---
{}
---
language: en
license: MIT
tags:
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

This model is based upon a BERT model that was fine-tuned
      on 30K pairs of texts.

- **Developed by:** Nigel Jose and Amitrajit Pati
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** [More Information Needed]
- **Paper or documentation:** [More Information Needed]

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

30K pairs of texts drawn from emails, news articles and blog posts.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 5 hours
      - duration per training epoch: 30 minutes
      - model size: 300MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 2K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 67% and an accuracy of 70%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
