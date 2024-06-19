#!/bin/bash

# Define the list of model IDs
model_ids=(
    "vinai/phobert-base"
    "vinai/phobert-base-v2"
    "uitnlp/visobert"
    "google-bert/bert-base-multilingual-uncased"
    "FacebookAI/xlm-roberta-base"
    "microsoft/deberta-v3-base"
    )

# Define the list of dataset types
dataset_types=("Phone" "Beauty" "Technology" "Mother" "Education", "Hotel", "Restaurant")
# "Restaurant" "Hotel" "Phone" "Beauty" "Technology" "Mother" "Education"

# Loop through each combination of model ID and dataset type
for model_id in "${model_ids[@]}"; do
  for domain in "${dataset_types[@]}"; do
    # Construct the Python script command with current model ID and dataset type
    python run_bert.py --model_id "$model_id" --domain "$domain"  --lr 2e-5 --num_epochs 10 --batch_size 32 --seed 42
    
  done
done