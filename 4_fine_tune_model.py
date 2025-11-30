

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_from_disk, ClassLabel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import logging
import gc

# --- Configuration ---
model_checkpoint = "google/muril-large-cased"
output_model_name = "punjabi-news-sentiment-muril-large"
checkpoints_dir = output_model_name + "_checkpoints"
saved_tokenized_dataset_path = "./tokenized_punjabi_news_muril_large"

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

# --- 1. Load Data ---
print(f"Loading data from {saved_tokenized_dataset_path}...")
try:
    tokenized_datasets = load_from_disk(saved_tokenized_dataset_path)
    print("✅ Dataset loaded.")
except Exception as e:
    print(f"❌ Error loading data: {e}. (Did you run Step 1 again after the crash?)")
    exit()

# --- 2. Label Mapping ---
label_feature = tokenized_datasets['train'].features['labels']
id2label = {i: name for i, name in enumerate(label_feature.names)}
label2id = {name: i for i, name in id2label.items()}
num_labels = len(id2label)

# --- 3. Load Model ---
print(f"Loading model: {model_checkpoint}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)

# --- 4. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1}

# --- 5. Class Weights ---
train_labels = np.array(tokenized_datasets["train"]["labels"])
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_labels),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 6. Training Arguments (STABILITY FOCUSED) ---
# Clean memory before starting
gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    learning_rate=2e-5,
    
    # --- CRITICAL SETTINGS FOR STABILITY ---
    per_device_train_batch_size=4,   # Reduced from 8 to 4
    per_device_eval_batch_size=8,    # Keep eval small
    gradient_accumulation_steps=8,   # Increased to 8 (Effective Batch = 32)
    gradient_checkpointing=True,     # SAVES MASSIVE VRAM (Must be True for Large models)
    dataloader_num_workers=0,        # SAVES SYSTEM RAM (Prevents the crash you saw)
    fp16=True,                       # Keep FP16 for speed
    # ---------------------------------------
    
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,
    report_to=[], 
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("--- Starting Training (Stability Mode) ---")
trainer.train()

print("--- Saving Final Model ---")
trainer.save_model(output_model_name + "_final")
tokenizer.save_pretrained(output_model_name + "_final")
    
print("\n--- Evaluation on Test Set ---")
metrics = trainer.evaluate(tokenized_datasets["test"])
print(metrics)
