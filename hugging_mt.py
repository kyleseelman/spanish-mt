from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

#model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model = AutoModelForSeq2SeqLM.from_pretrained("./results/5-epochs/checkpoint-25000")

generator = pipeline(task="translation", model=model, tokenizer=tokenizer)

#with open('mt_gender/data/aggregates/en.txt') as file:
#    data = ["" + line.strip() + "" for line in file]

df = pd.read_csv("./mt_gender/data/aggregates/en_anti.txt", sep="\t", names=["gender", "#", "text", "adj"])

print(df['text'])


#print(generator(text)[0]['translation_text'])
with open('translation_full_finetune_anti.txt', 'w') as f:

    for text in df['text']:
        #translations.append(model(text))
        translation = generator(text)[0]['translation_text']
        f.write(text + " ||| " + translation + '\n')
#        print(translation)
