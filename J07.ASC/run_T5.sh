#!/bin/bash

# Define the list of model IDs
model_ids=("google/mt5-base")

# Define the list of dataset types
dataset_types=("Phone" "Beauty" "Technology" "Mother" "Education")
# "Restaurant" "Hotel" "Phone" "Beauty" "Technology" "Mother" "Education"

# Loop through each combination of model ID and dataset type
for model_id in "${model_ids[@]}"; do
  for domain in "${dataset_types[@]}"; do
    # Construct the Python script command with current model ID and dataset type
    python script_T5.py --model_id "$model_id" --domain "$domain"  --lr 3e-4 --num_epochs 20 --batch_size 16 --seed 42
    
  done
done
