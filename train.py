import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. Load and preprocess data
print("Step 1: Loading data...")
df = pd.read_csv('data/IMDB_Dataset.csv')
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(f"Data loaded. Total samples: {len(df)}")

# 2. Split dataset
print("\nStep 2: Splitting data...")
train_df, test_df = train_test_split(
    df[['review', 'label']],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# 3. Convert to HuggingFace Dataset
dataset_train = Dataset.from_pandas(train_df.reset_index(drop=True))
dataset_eval = Dataset.from_pandas(test_df.reset_index(drop=True))

# 4. Load tokenizer and base model
print("\nStep 3: Loading model...")
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5. LoRA configuration
peft_config = LoraConfig(
    task_type='SEQ_CLS',
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)

base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

# 6. Tokenization function
def tokenize_batch(batch):
    return tokenizer(
        batch['review'],
        truncation=True,
        padding='max_length',
        max_length=128
    )

print("\nStep 4: Tokenizing data...")
tokenized_train = dataset_train.map(tokenize_batch, batched=True).remove_columns(['review']).with_format('torch')
tokenized_eval = dataset_eval.map(tokenize_batch, batched=True).remove_columns(['review']).with_format('torch')

# 7. Training arguments
training_args = TrainingArguments(
    output_dir='./outputs',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,
)

# 8. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, preds)}

# 9. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

print("\nStep 5: Training starts...")
trainer.train()

# 10. Evaluate
print("\nStep 6: Evaluating...")
metrics = trainer.evaluate()
print(f"Evaluation results: {metrics}")

# 11. Save model and tokenizer
print("\nStep 7: Saving model...")
model.save_pretrained('./lora_bert_imdb')
tokenizer.save_pretrained('./lora_bert_imdb')
