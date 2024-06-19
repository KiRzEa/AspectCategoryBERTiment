import pandas as pd
import os, random
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from preprocessing import *
from sklearn.metrics import *

def create_dataset(domain):
    path_train = "dataset/"+domain+ "_ABSA/csv/Train.csv"
    path_dev = "dataset/"+domain+ "_ABSA/csv/Dev.csv"
    path_test = "dataset/"+domain+ "_ABSA/csv/Test.csv"
    
    train = pd.read_csv(path_train)
    dev = pd.read_csv(path_dev)
    test = pd.read_csv(path_test)

    label2id = get_label2id(train)

    train['sentiment'] = train['sentiment'].map(label2id)
    dev['sentiment'] = dev['sentiment'].map(label2id)
    test['sentiment'] = test['sentiment'].map(label2id)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train),
        'dev': Dataset.from_pandas(dev),
        'test': Dataset.from_pandas(test)
    })

    return dataset, label2id 
    
def get_label2id(df):
    labels = df['sentiment'].unique().tolist()
    return {idx for idx, _ in labels}

def preprocess_function(examples, tokenizer, max_length, padding="max_length"):
    cleaned_review = clean_doc(examples['review'])
    tokenized_inputs = tokenizer(cleaned_review, examples['category'], max_length=max_length, padding=padding, truncation=True)
    tokenized_inputs['labels'] = examples['sentiment']

    return tokenized_inputs
    
def get_max_length(examples, tokenizer):
    return max([len(tokenizer(example['review'], example['category']).input_ids) for example in examples])

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=-1)

    accuracy_test = round(accuracy_score(labels, predictions)*100,4)
    balanced_accuracy = round(balanced_accuracy_score(labels, predictions)*100,4)
    f1_weighted_test = round(f1_score(labels, predictions, average='weighted')*100,4)
    f1_micro_test = round(f1_score(labels, predictions, average='micro')*100,4)
    f1_macro_test = round(f1_score(labels, predictions, average='macro')*100,4)

    return {
        'Acc': accuracy_test,
        'BalancedAcc': balanced_accuracy,
        'F1-Weighted': f1_weighted_test,
        'F1-Micro': f1_micro_test,
        'F1-Macro': f1_macro_test
    }

def evaluation_scores(y_test, y_pred,time_training,inference_time):
    accuracy_test = round(accuracy_score(y_test, y_pred)*100,4)
    balance_accuracy = round(balanced_accuracy_score(y_test, y_pred)*100,4)
    f1_weighted_test = round(f1_score(y_test, y_pred, average='weighted')*100,4)
    f1_micro_test = round(f1_score(y_test, y_pred, average='micro')*100,4)
    f1_macro_test = round(f1_score(y_test, y_pred, average='macro')*100,4)

    scores = "Accuracy: " + str(accuracy_test) + "\nBalance Accuracy: " + str(balance_accuracy) \
                + "\nWeighted F1-score: " + str(f1_weighted_test) \
                + "\nMacro F1-score: " + str(f1_macro_test) \
                + "\nMicro F1-score: " + str(f1_micro_test) \
                + "\nTraining time: " + str(time_training) \
                + "\nInference time: " + str(inference_time)
    
    scores += "\n" + str(classification_report(y_test, y_pred)) + "\n"
    print("===============\n")
    print(scores)
    return scores

def export_score_to_file(scores, model_id, domain):
    text_score = "Model: " + model_id +"\n Domain: " + domain + "\n" + scores + "\n\n"
    score_output_path = "scores/"+ "scores_" + str(model_id.replace("/", "_")) + ".txt"
    with open(score_output_path, 'a') as file:
        file.write(text_score)
    print("Save file done: ", score_output_path)
    

def save_prediction_to_file(review, category, y_true, y_pred, domain, model_id):
    path_output = "prediction/"+ str(domain) + "_" + str(model_id.replace("/", "_")) + ".csv"
    df = pd.DataFrame(list(zip(x_test, category, y_true, y_pred)), columns =['review', 'category', 'y_true', 'y_pred'])
    df.to_csv(path_output, index=False)
    
    return df

def set_seed(seed):
    """Set the seed for reproducibility in PyTorch."""
    random.seed(seed)            # Python random module.
    np.random.seed(seed)         # Numpy module.
    torch.manual_seed(seed)      # Sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for CUDA.
    torch.cuda.manual_seed_all(seed) # Sets the seed for all GPUs.
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set PYTHONHASHSEAT to prevent hash-based operations from randomness.
    
    # Configure PyTorch to be deterministic
    torch.backends.cudnn.deterministic = True  # Avoid nondeterministic algorithms.
    torch.backends.cudnn.benchmark = False     # If the input sizes for your neural network do not vary, turning off benchmarking can improve reproducibility.