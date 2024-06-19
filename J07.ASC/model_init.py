from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import default_data_collator, get_linear_schedule_with_warmup
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from preprocessing import *

def init_model_T5(model_id="google/mt5-large"):
    cache_dir = "/fs/scratch/Hc1_SX_AI-GPU-Admin/dan7hc/opinion"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir=cache_dir)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto",
                                                  trust_remote_code=True,
                                                  use_cache=False,
                                                  cache_dir=cache_dir,
                                                  force_download=True)
    return tokenizer, model


def init_trainer_T5(model, tokenizer, tokenized_dataset, learning_rate, batch_size, num_epochs):
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=16
    )

    # Define training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoint",
        #auto_find_batch_size= True,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate, # higher learning rate
        num_train_epochs=num_epochs,
        logging_dir="checkpoint/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )
    return trainer

# Get prediction
def model_T5_prediction(example, model, tokenizer,max_input_length,max_output_length):
    input_review = clean_doc(example, word_segment=False, lower_case=True)
    input_ids = tokenizer(input_review, max_length=max_input_length, return_tensors="pt", padding="max_length", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_output_length, num_beams=5)
    label = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    label = label.replace("<extra_id_0>", "").strip()
    return label


def init_model_LLM():
    return True