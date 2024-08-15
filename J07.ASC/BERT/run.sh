#!/bin/bash

# Define the list of model IDs
model_ids=(
    "vinai/phobert-base"
    "vinai/phobert-base-v2"
    "uitnlp/visobert"
    "FacebookAI/xlm-roberta-base"
    )

# Define the list of dataset types
dataset_types=("Phone" "Hotel" "Restaurant")
# "Restaurant" "Hotel" "Phone" "Beauty" "Technology" "Mother" "Education"

# Loop through each combination of model ID and dataset type
for model_id in "${model_ids[@]}"; do
  for domain in "${dataset_types[@]}"; do
    # Construct the Python script command with current model ID and dataset type
    python run_bert.py --model_id "$model_id" --domain "$domain"  --lr 2e-5 --num_epochs 10 --batch_size 16 --seed 8
    
  done
done