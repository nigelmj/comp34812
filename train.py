import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_scheduler
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configurations
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Custom Dataset
class NLIDataset(Dataset):
    def __init__(self, data):
        self.encodings = tokenizer(list(data['premise']), list(data['hypothesis']), padding=True, truncation=True, return_tensors="pt", max_length=512)
        self.labels = torch.tensor(data['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# DNN-Transformer Model
class DNNTransformerModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(DNNTransformerModel, self).__init__()
        self.transformer = BertModel.from_pretrained(model_name)
        self.dnn = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.dnn(hidden_state)
        return logits


# Load Data
train_data = pd.read_csv('/kaggle/input/trained/train.csv') 
val_data = pd.read_csv('/kaggle/input/trained/dev.csv')

# Create datasets
train_dataset = NLIDataset(train_data)
val_dataset = NLIDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize model
model = DNNTransformerModel(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Scheduler
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch {epoch + 1}/{EPOCHS}")
    for batch in tqdm(train_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(train_loader)}")

# Evaluation
model.eval()
predictions, true_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        logits = model(input_ids, attention_mask, token_type_ids)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        predictions.extend(preds)
        true_labels.extend(labels)

accuracy = accuracy_score(true_labels, predictions)
print(f"Validation Accuracy: {accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "dnn_transformer_nli.pth")