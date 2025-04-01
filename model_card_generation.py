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
    model_description = '''This model is based upon a BERT model that was fine-tuned
      on 30K pairs of texts.''',
    developers = 'Nigel Jose and Amitrajit Pati',
    base_model_repo = 'https://huggingface.co/google-bert/bert-base-uncased',
    base_model_paper = 'https://aclanthology.org/N19-1423.pdf',
    model_type = 'Supervised',
    model_architecture = 'Transformers',
    language = 'English',
    base_model = 'bert-base-uncased',

    # TODO: fill in the following attributes with the appropriate values

    training_data = '30K pairs of texts drawn from emails, news articles and blog posts.',
    hyperparameters = '''
      - learning_rate: 2e-05
      - train_batch_size: 16
      - eval_batch_size: 16
      - seed: 42
      - num_epochs: 10''',
    speeds_sizes_times = '''
      - overall training time: 5 hours
      - duration per training epoch: 30 minutes
      - model size: 300MB''',
    testing_data = 'A subset of the development set provided, amounting to 2K pairs.',
    testing_metrics = '''
      - Precision
      - Recall
      - F1-score
      - Accuracy''',
    results = 'The model obtained an F1-score of 67% and an accuracy of 70%.',
    hardware_requirements = '''
      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100''',
    software = '''
      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113''',
    bias_risks_limitations = '''Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.''',
    additional_information = '''The hyperparameters were determined by experimentation
      with different values.'''
)

# the following lines will write a markdown (.md) file; this becomes one of your model cards
# change the filename accordingly
with open('natural-language-inference-transformer-model-card.md', 'w') as model_card:
  model_card.write(card.content)