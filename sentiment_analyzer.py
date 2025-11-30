from google.colab import drive
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
import glob
import sys

# 1. Mount Drive
drive.mount('/content/drive')

# --- Configuration ---
DRIVE_ROOT = "/content/drive/MyDrive/Punjabi_Sentiment_Project"
CHECKPOINT_FOLDER = "punjabi-news-sentiment-l3cube_checkpoints"
BASE_PATH = f"{DRIVE_ROOT}/{CHECKPOINT_FOLDER}"

# --- Find Backup ---
print(f"🔎 Looking inside: {BASE_PATH} ...")

if not os.path.exists(BASE_PATH):
    print("❌ Error: Checkpoint folder not found.")
    sys.exit()

subfolders = glob.glob(f"{BASE_PATH}/checkpoint-*")
if not subfolders:
    print("❌ Error: No checkpoints found!")
    sys.exit()

# Get the latest checkpoint
latest_model_path = max(subfolders, key=lambda p: int(p.split('-')[-1]))
print(f"✅ FOUND BACKUP MODEL: {latest_model_path}")

# --- Load Model ---
print("🚀 Loading model (Attempt 2)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(latest_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(latest_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("✅ Success! Model loaded.")

except AttributeError:
    print("❌ Library Bug Detected: Please run '!pip install transformers==4.57.1' and Restart Runtime.")
except Exception as e:
    print(f"❌ Critical Error: {e}")
    sys.exit()

# --- Interactive Loop ---
while True:
    text = input("\n👉 Enter Headline (or 'exit'): ")
    if text.lower() in ['exit', 'quit']: break
    if not text.strip(): continue

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        sentiment = model.config.id2label[pred_idx]

    print(f"📊 Result: {sentiment} ({confidence:.1%})")
