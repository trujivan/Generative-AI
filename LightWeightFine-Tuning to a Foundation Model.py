from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# CODE 1: Loading and Evaluating a Foundation Model

# Load an appropriate dataset
dataset_name = "poem_sentiment"  # Replace with the actual dataset name
dataset = load_dataset(dataset_name, split="train[:150]")
#To ensure that this example runs within a reasonable time frame, here we are limiting the number of instances from the training set of the SceneParse150 dataset to 150.

#spliting the dataset into train and test sets.
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Load the foundation model
PoemSentModel = "gpt2"
foundation_model = AutoModelForCausalLM.from_pretrained(PoemSentModel)
tokenizer = AutoTokenizer.from_pretrained(PoemSentModel)

# Example: Tokenize and evaluate
input_text = dataset['train'][0]['text']  # Use any text example from your dataset
inputs = tokenizer(input_text, return_tensors="pt")
foundation_outputs = foundation_model(**inputs)

# Perform any necessary evaluation based on your task

# CODE 2: Performing Parameter-Efficient Fine-Tuning (PEFT)

# Create a PEFT config
peft_config = LoraConfig()  # You can adjust this based on your needs

# Convert the foundation model into a PEFT model
peft_model = get_peft_model(foundation_model, peft_config)

# Assuming you have a training loop or Hugging Face Trainer for fine-tuning
# Make sure you replace this with your actual fine-tuning code and dataset
# Here, we're using a placeholder variable 'peft_dataset'
peft_dataset = load_dataset("your/fine_tuning_dataset")
# Replace this with your fine-tuning code
for epoch in range(1):
    for batch in peft_dataset['train']:
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True)
        outputs = peft_model(**inputs, labels=batch['label'])
        # Your fine-tuning training step

# Save the trained PEFT model
peft_model.save_pretrained("gpt-peft")

# Performing Inference with a PEFT Model

# Load the PEFT model
peft_model = AutoModelForCausalLM.from_pretrained("gpt-peft")

# Tokenize and evaluate with the PEFT model
input_text = dataset['train'][0]['text']  # Use any text example from your dataset
inputs = tokenizer(input_text, return_tensors="pt")
peft_outputs = peft_model(**inputs)
# Perform any necessary evaluation based on your task

# You can now compare foundation_outputs and peft_outputs to observe the impact of PEFT
