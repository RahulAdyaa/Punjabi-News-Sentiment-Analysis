import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

input_file = 'punjabi_news_sentiment_labelled_dataset_finall_3june.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    sys.exit()

try:
    df = pd.read_csv(input_file, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, encoding='latin1')

df = df.dropna(subset=['Headline', 'Sentiment'])

# Split 1: 10% Test
train_val, test = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df['Sentiment']
)

# Split 2: 10% Validation (0.1111 of the remaining 90%)
train, val = train_test_split(
    train_val,
    test_size=0.1111,
    random_state=42,
    stratify=train_val['Sentiment']
)

train.to_csv('train_set.csv', index=False)
val.to_csv('validation_set.csv', index=False)
test.to_csv('test_set.csv', index=False)

print(f"Train rows: {len(train)}")
print(f"Val rows:   {len(val)}")
print(f"Test rows:  {len(test)}")
