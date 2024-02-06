# PyTorch and HuggingFace Scavenger Hunt

#Hello, GitHub community! 👋 I'm on a journey to explore the powerful tools of PyTorch and HuggingFace. Join me in this scavenger hunt to uncover hidden treasures along the way.

## Part 1: Familiarize Yourself with PyTorch

### PyTorch Tensors
#Let's start by creating a PyTorch tensor named `my_tensor`. Ensure it's of size 3x3 with values of our choice, and created on the GPU if available.

import torch

# Set the device to be used for the tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a tensor on the appropriate device
my_tensor = torch.randn((3, 3), device=device)

# Print the tensor
print(my_tensor)

# Check the device and shape
assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3, 3)

print("Success!")


### Neural Net Constructor Kit `torch.nn` 
#Now, let's construct a three-layer Multi-Layer Perceptron (MLP) using PyTorch's `torch.nn` module.

import torch.nn as nn

class MyMLP(nn.Module):
    """My Multilayer Perceptron (MLP)

    Specifications:

        - Input layer: 784 neurons
        - Hidden layer: 128 neurons with ReLU activation
        - Output layer: 10 neurons with softmax activation

    """
    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

my_mlp = MyMLP()
print(my_mlp)

# Check the architecture
assert my_mlp.fc1.in_features == 784
assert my_mlp.fc2.out_features == 10
assert my_mlp.fc1.out_features == 128
assert isinstance(my_mlp.fc1, nn.Linear)
assert isinstance(my_mlp.fc2, nn.Linear)

### PyTorch Loss Functions and Optimizers 
#Let's create a loss function using `torch.nn.CrossEntropyLoss` and an optimizer using `torch.optim.SGD`.

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(my_mlp.parameters(), lr=0.001)

# Check the types and settings
assert isinstance(loss_fn, nn.CrossEntropyLoss)
assert isinstance(optimizer, torch.optim.SGD)
assert optimizer.defaults["lr"] == 0.001
assert optimizer.param_groups[0]["params"] == list(my_mlp.parameters())

### PyTorch Training Loops
#PyTorch makes writing a training loop easy!

def fake_training_loaders():
    for _ in range(30):
        yield torch.randn(64, 784), torch.randint(0, 10, (64,))

for epoch in range(3):
    # Create a training loop
    for i, data in enumerate(fake_training_loaders()):
        x, y = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass (predictions)
        y_pred = my_mlp(x)

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, batch {i}: {loss.item():.5f}")

## Part 2: Get to Know HuggingFace

### Download a Model from HuggingFace for Sentiment Analysis
#Let's use the `distilbert-base-uncased-finetuned-sst-2-english` model for sentiment analysis.

from transformers import AutoModelForSequenceClassification, AutoTokenizer

pt_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def get_prediction(review):
    inputs = tokenizer(review, return_tensors="pt")
    outputs = pt_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return "positive" if predictions.item() == 1 else "negative"

# Check sentiment predictions
review_negative = "This movie is not so great :("
print(f"Review: {review_negative}")
print(f"Sentiment: {get_prediction(review_negative)}")

assert get_prediction(review_negative) == "negative", "The prediction should be negative"

review_positive = "This movie rocks!"
print(f"Review: {review_positive}")
print(f"Sentiment: {get_prediction(review_positive)}")

assert get_prediction(review_positive) == "positive", "The prediction should be positive"

### Download a Dataset from HuggingFace
#Let's use the IMDb dataset for our