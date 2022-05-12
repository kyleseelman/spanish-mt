from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datasets

from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

#books = load_dataset("opus_books", "en-fr")
#print(books)

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")

data = np.load('fine_tune_full_data.npy',allow_pickle=True)
data = data.tolist()
#train_data, test_data = train_test_split(data, test_size=0.2)
#print(train_data[0]['translation'])
#print(test_data[0]['translation']['es'])
print(pd.DataFrame(data=data))
data = datasets.Dataset.from_pandas(pd.DataFrame(data=data))
print(data)
data = data.train_test_split(test_size=0.2)


#def tokenize_function(examples):
#    return tokenizer(examples, padding="max_length", truncation=True)

#tokenized_datasets = data.map(tokenize_function, batched=True)

#train_dataset = tokenized_datasets["train"].shuffle(seed=42)
#eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

source_lang = "en"
target_lang = "es"
prefix = "translate English to Spanish: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#inputs = [example for example in train_data[0]['translation']]
#targets = [example for example in train_data[0]['translation']]
#print(inputs)
#print(targets)

tokenized_books = data.map(preprocess_function, batched=True)
print(tokenized_books)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


training_args = Seq2SeqTrainingArguments(
    output_dir="./results/5-epochs",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()