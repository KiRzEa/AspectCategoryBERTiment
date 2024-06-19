import os
import argparse
import time
from utils import *
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

#======================================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
#======================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="vinai/phobert-base")
parser.add_argument("--domain", type=str, default="Restaurant")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

model_id = args.model_id
domain = args.domain
learning_rate = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
seed = args.seed
#======================================
print("="*50)
print("[INFO] Model ID: ", model_id)
print("[INFO] Type of Dataset: ", domain)
print("[INFO] Learning Rate: ", learning_rate)
print("[INFO] Number of Epochs: ", num_epochs)
print("[INFO] Batch Size: ", batch_size)
print("[INFO] Seed: ", seed)
print("="*50)
#======================================
set_seed(seed)
#======================================
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
#======================================
dataset, label2id = create_dataset(domain)
#======================================
max_length = get_max_length(dataset['train'], tokenizer)
#======================================
tokenized_dataset = dataset.map(preprocess_function, 
                                fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length},
                                batched=True,
                                batch_size=1024,
                                remove_columns=dataset['train'].column_names)
#======================================
training_args = TrainingArguments(
    output_dir="checkpoint",
    #auto_find_batch_size= True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size*2,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    logging_dir="checkpoint/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
)

collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    compute_metrics=compute_metrics
)
#======================================
start_time= time.time()
trainer.train()
stop_time=time.time()
time_training =stop_time - start_time
print("Training time (seconds): ", time_training)
#======================================
start_time= time.time()
predictions, labels, metrics = trainer.predict(tokenized_dataset['test'])
stop_time=time.time()
inference_time =stop_time - start_time
print("Inference time (seconds): ", inference_time)
#======================================
# Evaluation

scores = evaluation_scores(labels, predictions, time_training, inference_time)
export_score_to_file(scores, model_id, domain)
save_prediction_to_file(dataset['test']['review'], dataset['test']['category'], y_test, y_pred, domain, model_id)

#======================================