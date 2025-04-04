from huggingface_hub import ModelCard, ModelCardData

card_data = ModelCardData(
    language='en',
    license='MIT',
    tags=[
        'sequence-classification',
        'pairwise-classification',
        'natural-language-inference',
        'esim',
        'ensemble',
        'nlu',
    ],
    repo="https://github.com/nigelmj/comp34812",
    ignore_metadata_errors=True)

card = ModelCard.from_template(
    card_data = card_data,
    template_path='COMP34812_modelcard_template.md',
    model_id = 'p64932nj-y24592ap-NLI',
    model_summary = '''This is a classification model that, given a premise and a hypothesis,
      was trained to determine whether the hypothesis is true based on the premise.''',
    model_description = '''This model is based on ESIM-inspired attention and local inference modeling.
      It was trained as an ensemble of LSTM, BiLSTM, GRU, and BiGRU architectures, with pretrained GloVe word embeddings.''',
    developers = 'Nigel Jose and Amitrajit Pati',
    model_type = 'Supervised',
    language = 'English',

    base_model_repo = 'N/A',
    base_model_paper = '''
    - [Ensemble Deep Learning on Time-Series Representation of Tweets for Rumor Detection in Social Media](https://arxiv.org/pdf/2004.12500)
    - [Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038)
    ''',
    model_architecture = 'RNN Ensemble with Attention and Local Inference Modelling',

    training_data = '24K pairs of texts provided by external party.',
    hyperparameters = '''
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
      - num_epochs: 20''',
    speeds_sizes_times = '''
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
        * BiGRU: 40MB''',
    testing_data = 'Model has been tested on the validation set (6K pairs). Further testing on an external test set is ongoing.',
    testing_metrics = '''
      - Precision
      - Recall
      - F1-score
      - Accuracy''',
    results = 'The model obtained an F1-score of 71% and an accuracy of 71% on the validation set.',
    hardware_requirements = '''
      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: P100''',
    software = '''
      - TensorFlow 2.17.1 (CUDA 12.3, cuDNN 8)''',
    bias_risks_limitations = '''The model was trained with a max sequence length of 35 tokens for premise and hypothesis, corresponding to the 95th percentile of the dataset.
      The dataset was provided and not collected by the authors. No explicit data collection methodology is available.
      The random seed was not set, as GPU-based training may lead to non-deterministic behavior despite the presence of an explicit seed setting.''',
    additional_information = '''The hyperparameters were determined by experimentation
      with different values.'''
)

# the following lines will write a markdown (.md) file; this becomes one of your model cards
# change the filename accordingly
with open('natural-language-inference-model-card.md', 'w') as model_card:
  model_card.write(card.content)