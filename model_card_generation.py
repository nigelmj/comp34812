from huggingface_hub import ModelCard, ModelCardData

card_data = ModelCardData(
    language='en',
    license='MIT',
    tags=[
        'transformer',
        'sequence-classification',
        'pairwise-classification',
        'natural-language-inference',
        'nlp',
    ],
    repo="https://github.com/nigelmj/comp34812",
    ignore_metadata_errors=True)

card = ModelCard.from_template(
    card_data = card_data,
    template_path='COMP34812_modelcard_template.md',
    model_id = 'p64932nj-y24592ap-NLI',
    model_summary = '''This is a classification model that, given a premise and a hypothesis,
      was trained to determine whether the hypothesis is true based on the premise.''',
    model_description = '''This model is based upon a RoBerta model that was fine-tuned
      on 30K pairs of texts + synthetically generated texts.''',
    developers = 'Nigel Jose and Amitrajit Pati',
    base_model_repo = 'https://huggingface.co/google-bert/bert-base-uncased',
    base_model_paper = 'https://aclanthology.org/N19-1423.pdf',
    model_type = 'Supervised',
    model_architecture = 'Transformers',
    language = 'English',
    base_model = 'roberta-base',
    finetuned_from_model = 'roberta-base'

    # TODO: fill in the following attributes with the appropriate values

    training_data = '24K pairs of texts from train.csv and synthetic data created from the file',
    hyperparameters = '''
      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 1
      - warmup_ratio: 0.1
      - weight_decay: 0.01''',
    speeds_sizes_times = '''
      - overall training time: 40 minutes
      - duration per training epoch: 15 minutes
      - model size: 500.78MB''',
    testing_data = 'A testing file provided to us, called dev.csv'
    testing_metrics = '''
      - Precision = 0.86162742994469
      - Recall = 0.86163895486936
      - F1-score = 0.86163045534955
      - Accuracy = 0.86163895486936
      ''',
    results = 'The model obtained an F1-score of 86% and an accuracy of 86%.',
    hardware_requirements = '''
      - RAM: at least 16 GB
      - Storage: at least 1GB,
      - GPU: P100''',
    software = '''
      - Transformers 4.47.0
      - Pytorch 2.5.1+cu121
      -Cuda 12.1
      -CuDNN 9''',
    bias_risks_limitations = '''Any inputs (concatenation of two sequences) longer than
      256 subwords will be truncated by the model.
      The dataset was provided and not collected by the authors. No explicit data collection methodology is available. 
      The random seed is set, but GPU-based training may lead to non-deterministic behavior despite the presence of an explicit seed setting.
      The augmented data creation uses a simple algorithm and might cause the model to hallucinate''',
    additional_information = '''The hyperparameters were determined by experimentation
      with different values. And additional data addition technique was from the paper :Arxiv.org. (2015). Improving the Natural Language Inference robustness to hard dataset by data augmentation and preprocessing. [online] Available at: https://arxiv.org/html/2412.07108v1#S3.'''
)

# the following lines will write a markdown (.md) file; this becomes one of your model cards
# change the filename accordingly
with open('natural-language-inference-transformer-model-card.md', 'w') as model_card:
  model_card.write(card.content)