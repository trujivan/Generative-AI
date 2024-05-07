from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer

# CODE 1: Loading and Evaluating a Foundation Model

# Load an appropriate dataset
dataset_name = "poem_sentiment"  # Replace with the actual dataset name
dataset = load_dataset(dataset_name, split="train[:150]")

# Split the dataset into train and test sets
train_dataset, test_dataset = train_test_split(dataset['train'], test_size=0.1, random_state=42)

# Load the foundation model
PoemSentModel = "gpt2"
foundation_model = AutoModelForSequenceClassification.from_pretrained(PoemSentModel)
tokenizer = AutoTokenizer.from_pretrained(PoemSentModel)

# Example: Tokenize and evaluate
input_text = train_dataset[0]['text']  # Use any text example from your dataset
inputs = tokenizer(input_text, return_tensors="pt")
foundation_outputs = foundation_model(**inputs)

# Perform any necessary evaluation based on your task

# CODE 2: Performing Parameter-Efficient Fine-Tuning (PEFT)

# Create a PEFT config
peft_config = LoraConfig()  # You can adjust this based on your needs

# Convert the foundation model into a PEFT model
peft_model = get_peft_model(foundation_model, peft_config)

# Fine-tuning setup
training_args = TrainingArguments(
    output_dir="./peft_output",
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    save_steps=500,
)

# Define a simple training function
def fine_tune_peft_model(model, training_args, train_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

# Assuming you have a training loop or Hugging Face Trainer for fine-tuning
# Replace this with your actual fine-tuning code and dataset
peft_train_dataset, peft_test_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

# Fine-tune the PEFT model
fine_tune_peft_model(peft_model, training_args, peft_train_dataset)

# Save the trained PEFT model
peft_model.save_pretrained("gpt-lora")

# Performing Inference with a PEFT Model

# Load the PEFT model
peft_model = AutoModelForSequenceClassification.from_pretrained("gpt-lora")

# Tokenize and evaluate with the PEFT model
input_text = test_dataset[0]['text']  # Use any text example from your test dataset
inputs = tokenizer(input_text, return_tensors="pt")
peft_outputs = peft_model(**inputs)
# Perform any necessary evaluation based on your task

# You can now compare foundation_outputs and peft_outputs to observe the impact of PEFT










Certainly! Below is the modified code using PyTorch for fine-tuning and inference:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import AdamW

# CODE 1: Loading and Evaluating a Foundation Model

# Load an appropriate dataset
dataset_name = "poem_sentiment"  # Replace with the actual dataset name
dataset = load_dataset(dataset_name, split="train[:150]")

# Split the dataset into train and test sets
train_dataset, test_dataset = train_test_split(dataset['train'], test_size=0.1, random_state=42)

# Load the foundation model
PoemSentModel = "gpt2"
foundation_model = AutoModelForSequenceClassification.from_pretrained(PoemSentModel)
tokenizer = AutoTokenizer.from_pretrained(PoemSentModel)

# Example: Tokenize and evaluate
input_text = train_dataset[0]['text']  # Use any text example from your dataset
inputs = tokenizer(input_text, return_tensors="pt")
foundation_outputs = foundation_model(**inputs)

# Perform any necessary evaluation based on your task

# CODE 2: Performing Parameter-Efficient Fine-Tuning (PEFT)

# Create a PEFT config
peft_config = LoraConfig()  # You can adjust this based on your needs

# Convert the foundation model into a PEFT model
peft_model = get_peft_model(foundation_model, peft_config)

# Fine-tuning setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
peft_model.to(device)

# Convert datasets to PyTorch DataLoader
def convert_to_dataloader(dataset):
    inputs = tokenizer(dataset['text'], return_tensors="pt", padding=True, truncation=True)
    labels = torch.tensor(dataset['label'].tolist())
    return DataLoader(TensorDataset(**inputs, labels=labels), batch_size=8, shuffle=True)

peft_train_dataloader = convert_to_dataloader(train_dataset)
peft_test_dataloader = convert_to_dataloader(test_dataset)

# Fine-tune the PEFT model
optimizer = AdamW(peft_model.parameters(), lr=1e-5)
num_epochs = 1
for epoch in range(num_epochs):
    peft_model.train()
    for batch in peft_train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = peft_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the trained PEFT model
peft_model.save_pretrained("gpt-peft")

# Performing Inference with a PEFT Model

# Load the PEFT model
peft_model = AutoModelForSequenceClassification.from_pretrained("gpt-peft").to(device)

# Tokenize and evaluate with the PEFT model
input_text = test_dataset[0]['text']  # Use any text example from your test dataset
inputs = tokenizer(input_text, return_tensors="pt").to(device)
peft_outputs = peft_model(**inputs)

# Perform any necessary evaluation based on your task

# You can now compare foundation_outputs and peft_outputs to observe the impact of PEFT
```

