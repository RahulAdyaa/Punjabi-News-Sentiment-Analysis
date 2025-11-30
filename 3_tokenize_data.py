
# STEP 1: TOKENIZATION SCRIPT (L3Cube-Punjabi-BERT)
# Saves output to Google Drive for safety.

from google.colab import drive
import pandas as pd
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
import logging
import os
import shutil

# --- 1. Mount Google Drive ---
drive.mount('/content/drive')

# --- Configuration ---
# Input CSVs (These should be in your Colab Files tab from Step 0)
train_csv = 'train_set.csv'
validation_csv = 'validation_set.csv'
test_csv = 'test_set.csv'

# Column Headers
text_column = 'Headline'
label_column = 'Sentiment'

# MODEL: L3Cube Punjabi BERT
model_checkpoint = "l3cube-pune/punjabi-bert"


max_token_length = 256


DRIVE_ROOT = "/content/drive/MyDrive/Punjabi_Sentiment_Project"
saved_tokenized_dataset_path = f"{DRIVE_ROOT}/tokenized_l3cube"

# --- Logging & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

# Clear old cache if it exists to ensure a fresh start
if os.path.exists(saved_tokenized_dataset_path):
    print(f"🧹 Clearing old data at {saved_tokenized_dataset_path}...")
    shutil.rmtree(saved_tokenized_dataset_path)

# --- 2. Load Data ---
print("\n--- Loading Data ---")
if not all(os.path.exists(f) for f in [train_csv, validation_csv, test_csv]):
    print("❌ Error: Input CSV files not found.")
    print("   -> Did you run Step 0 (Splitting) after the restart?")
    exit()

try:
    raw_datasets = load_dataset('csv', data_files={'train': train_csv, 'validation': validation_csv, 'test': test_csv})

    # Filter out empty rows just in case
    for split in raw_datasets:
        raw_datasets[split] = raw_datasets[split].filter(
            lambda x: x[label_column] is not None and x[text_column] is not None
        )
    print("✅ Data Loaded Successfully.")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- 3. Prepare Labels ---
print("\n--- Processing Labels ---")
try:

    unique_labels = sorted(list(set(str(lbl).strip() for lbl in raw_datasets['train'][label_column])))
    class_label_feature = ClassLabel(names=unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}

    print(f"Labels found: {unique_labels}")

    def map_labels(batch):
        batch['labels'] = [label2id.get(str(l).strip(), -1) for l in batch[label_column]]
        return batch

    raw_datasets = raw_datasets.map(map_labels, batched=True)

    new_features = raw_datasets['train'].features.copy()
    new_features['labels'] = class_label_feature
    raw_datasets = raw_datasets.cast(new_features)

except Exception as e:
    print(f"❌ Error preparing labels: {e}")
    exit()

# --- 4. Tokenization ---
print(f"\n--- Tokenizing with {model_checkpoint} ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        text_batch = [str(t) for t in examples[text_column]]
        return tokenizer(text_batch, padding="max_length", truncation=True, max_length=max_token_length)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    
    cols_to_remove = [c for c in tokenized_datasets['train'].column_names if c not in ['input_ids', 'attention_mask', 'labels']]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    tokenized_datasets.set_format("torch")

except Exception as e:
    print(f"❌ Error during tokenization: {e}")
    exit()

# --- 5. Save to Google Drive ---
print(f"\n💾 Saving processed data to: {saved_tokenized_dataset_path} ...")
try:
    tokenized_datasets.save_to_disk(saved_tokenized_dataset_path)
    print("✅ SUCCESS! Data saved to Drive.")
    print("👉 You can now run Step 2 (Fine-Tuning).")
except Exception as e:
    print(f"❌ Error saving to Drive: {e}")
