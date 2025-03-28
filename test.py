import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configurations (should match training script)
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset (same as in training script)
class NLIDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.encodings = tokenizer(list(data['premise']), list(data['hypothesis']), 
                                   padding=True, truncation=True, 
                                   return_tensors="pt", max_length=512)
        self.labels = torch.tensor(data['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Load the model class (same as in training script)
class DNNTransformerModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(DNNTransformerModel, self).__init__()
        self.transformer = BertModel.from_pretrained(model_name)
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(self.transformer.config.hidden_size, 512),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.dnn(hidden_state)
        return logits

def test_model(dev_path='/kaggle/input/testing/NLI_trial.csv', model_path='/kaggle/working/dnn_transformer_nli.pth'):
    # Load dev data
    dev_data = pd.read_csv(dev_path)
    
    # Create dev dataset
    dev_dataset = NLIDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    
    # Initialize and load model
    model = DNNTransformerModel(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()
    
    # Prediction
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            token_type_ids = batch['token_type_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = labels.cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)
    
    # Generate detailed reports
    print("Classification Report:")
    print(classification_report(true_labels, predictions))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # Optional: Save predictions to CSV
    results_df = dev_data.copy()
    results_df['predicted_label'] = predictions
    results_df['is_correct'] = (results_df['label'] == results_df['predicted_label'])
    results_df.to_csv('dev_predictions.csv', index=False)
    
    return predictions, true_labels

# Run the test
if __name__ == "__main__":
    test_model()