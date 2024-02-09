import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Assuming you have a task-specific dataset with 'text' and 'label' columns
# Load your dataset and split it into training and testing sets
# For simplicity, let's assume you have a DataFrame called 'df'
# with columns 'text' and 'label' where 'label' represents your task-specific labels

# Example code for loading your dataset
# Replace this with loading your actual dataset
df = your_task_specific_dataset_loading_function()

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define a custom dataset class
class YourTaskDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        return {'text': text, 'label': label}

# Create datasets and data loaders
train_dataset = YourTaskDataset(train_df['text'].values, train_df['label'].values)
test_dataset = YourTaskDataset(test_df['text'].values, test_df['label'].values)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}')

# Evaluation on the test set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = batch['label'].to(device)

        outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy:.4f}')
