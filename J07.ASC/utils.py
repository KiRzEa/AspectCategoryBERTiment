import pandas as pd
import re, os, string, random
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from preprocessing import *
from category_mapping import *

def read_dataset(domain):
    path_train = "dataset/"+domain+ "_ABSA/csv/Train.csv"
    path_dev = "dataset/"+domain+ "_ABSA/csv/Dev.csv"
    path_test = "dataset/"+domain+ "_ABSA/csv/Test.csv"
    
    df_train = pd.read_csv(path_train)
    df_dev = pd.read_csv(path_dev)
    df_test = pd.read_csv(path_test)
    return df_train,df_dev,df_test

def mapping_label2vietnamese(label):
    index = ["negative","neutral", "positive"].index(label)
    return ["tệ", "tạm", "tốt"][index]

def extract_input_output_for_T5(df, domain, type_dataset = "train"):
    reviews = df["review"].tolist()
    categories = df["category"].tolist()
    sentiments = df["sentiment"].tolist()
    
    intput_list = []
    output_list = []
    prompt = "Hãy phân loại trạng thái cảm xúc của "
    for index,review in enumerate(reviews):
        review_clean = clean_doc(review, word_segment=False, lower_case=True)
        text_prompt = "Hãy phân loại trạng thái cảm xúc của khía cạnh: \"" + mapping_category(domain,categories[index]) + "\" trong bình luận sau: " + review_clean
        intput_list.append(text_prompt)
        output_list.append(mapping_label2vietnamese(sentiments[index]))
    print(len(intput_list),len(output_list))
    
    df_format = pd.DataFrame(list(zip(intput_list, output_list)), columns =['x_input', 'y_output'])
    dataset_asc = DatasetDict()
    if type_dataset == "train":
        dataset_asc['train'] = Dataset.from_pandas(df_format)
        return dataset_asc
    else:
        return intput_list, output_list


def preprocess_function_T5(df,tokenizer,max_input_length,max_output_length,padding="max_length"):
    text_column = "x_input"
    label_column = "y_output"
    
    model_inputs = tokenizer(text_target=df[text_column], max_length=max_input_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=df[label_column], max_length=max_output_length, padding=padding, truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)
    
def get_max_input_output(tokenizer,train_df):
    input_encodings = tokenizer(train_df["x_input"])
    output_encodings = tokenizer(train_df["y_output"])
    max_input = find_max_list(input_encodings["input_ids"])
    max_output = find_max_list(output_encodings["input_ids"])
    return max_input, max_output

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