from preprocessing import *
from utils import *
from model_init import *
from transformers import DataCollatorForSeq2Seq
import time, os, warnings
import argparse
from evaluation_script import *

#======================================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
#======================================

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="google/mt5-large")
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
# Init tokenizer, model
tokenizer,model = init_model_T5(model_id)

#======================================
df_train,df_dev,df_test = read_dataset(domain)
training_dataset = extract_input_output_for_T5(df_train,domain,type_dataset="train")
#======================================
max_input_length, max_output_length = get_max_input_output(tokenizer,training_dataset['train'])

#======================================
tokenized_dataset = training_dataset.map(preprocess_function_T5, 
                                        fn_kwargs={'max_input_length': max_input_length, 'max_output_length': max_output_length,
                                              'tokenizer': tokenizer},
                                        batched=True, remove_columns=["x_input", "y_output"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

#======================================

#======================================
# Init Trainer 
trainer = init_trainer_T5(model, tokenizer, tokenized_dataset, learning_rate, batch_size, num_epochs)
model.config.use_cache = False

#======================================
start_time= time.time()
trainer.train()
stop_time=time.time()
time_training =stop_time - start_time
print("Training time (seconds): ", time_training)
#======================================
# Prediction
#======================================
x_test,y_test = extract_input_output_for_T5(df_test,domain,type_dataset="test")

start_time= time.time()
y_pred = []
for text in x_test:
    pred = model_T5_prediction(text, model, tokenizer,max_input_length,max_output_length)
    y_pred.append(pred)
stop_time=time.time()
inference_time =stop_time - start_time
print("Inference time (seconds): ", inference_time)

#======================================
# Evaluation


scores = evaluation_scores(y_test, y_pred,time_training,inference_time)
export_score_to_file(scores, model_id, domain)
save_prediction_to_file(x_test, y_test, y_pred, domain, model_id)

#======================================