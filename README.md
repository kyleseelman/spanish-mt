# Mitigating Gender Bias in Spanish Translations

##### These scripts are still in development, so the code is currently not modular (sorry!) each file needs to be adapted to point to the correct directory for all models trained/data created throughout this process
##### Download Europarl dataset from: https://www.statmt.org/europarl/ and change data path in code to location of data
##### Run 'python3 -m spacy download es_core_news_md' to download Spanish parser from spaCy
##### Run 'pip install -r requirements.txt' to install all necessary dependencies
##### Run 'python europarl.py' to generate the gender-balanced dataset of 100k samples. It also outputs pronoun proportions of before and after. (Takes a while)
##### Run 'python combine_data.py' to generate the counterfactuals for English and combine the English and Spanish translations into a single dataset
##### Run 'python fine-tune.py' to fine-tune the Hugging Face NMT model (https://huggingface.co/Helsinki-NLP/opus-mt-en-es) on the generated gender-balanced dataset
##### Now the fine-tuned model is created we want to evaluate on WinoMT following: https://github.com/gabrielStanovsky/mt_gender
##### Run 'python hugging_mt.py' will generate the translations that are needed for WinoMT. Then follow the instructions on their Github to evaluate.

##### If you want to use the gender-balanced dataset to train a Hugging Face NMT from scratch, run 'python scratch.py'. The hyperparameters are just set to default, but can certaintly be optimized
