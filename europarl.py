import pandas as pd
import numpy as np
import re
import numpy as np

with open('./data/europarl-v7.es-en.en', encoding="utf-8") as file:
        # Read the line from file, strip leading and trailing whitespace,
        # prepend the start-text and append the end-text.
        texts_en = ["" + line.strip() + "" for line in file]

with open('./data/europarl-v7.es-en.es', encoding="utf-8") as file:
        # Read the line from file, strip leading and trailing whitespace,
        # prepend the start-text and append the end-text.
        texts_es = ["" + line.strip() + "" for line in file]
# 1,965,734
#print(len(texts_en))
#print(len(texts_es))

male_articles = ['el', 'él', 'los', 'un', 'unos', 'estos', 'aquello', 'aquellos', 'ellos']
female_articles = ['la', 'las', 'una', 'unas', 'estas', 'aquella', 'aquellas', 'ella', 'ellas']


temp = texts_es
counter_ellos = 0
counter_ellas = 0
counter_el = 0
counter_ella = 0
counter_nosotros = 0
counter_nosotras = 0
counter_vosotros = 0
counter_vosotras = 0
male_counter = 0
female_counter = 0
for line in temp[0:100000]:
    if " ellos " in line or "Ellos" in line: counter_ellos += 1 #print(line.replace('ellos', 'ellas'))
    if " ellas " in line or "Ellas" in line: counter_ellas += 1
    if " él " in line or "Él" in line: counter_el += 1
    if " ella " in line or "Ella" in line: counter_ella +=1
    if " nosotros " in line or "Nosotros" in line: counter_nosotros +=1
    if " nosotras " in line or "Nosotras" in line: counter_nosotras +=1
    if " vosotros " in line or "Vosotros" in line: counter_vosotros +=1
    if " vosotras " in line or "Nosotras" in line: counter_vosotras +=1
    #if any([word in line for word in male_articles]): male_counter += 1
    #if any([word in line for word in female_articles]): female_counter += 1

print("Number of ellos occurences: ", counter_ellos)
print("Number of ellas occurences: ", counter_ellas)
print("Number of el occurences: ", counter_el)
print("Number of ella occurences: ", counter_ella)
print("Number of nosotros occurences: ", counter_nosotros)
print("Number of nosotras occurences: ", counter_nosotras)
print("Number of vosotros occurences: ", counter_vosotros)
print("Number of vosotras occurences: ", counter_vosotras)
print("Percentage of male pronouns: ", (counter_el+counter_ellos+counter_nosotros+counter_vosotros)/(counter_el+counter_ella+counter_ellas+counter_ellos+counter_nosotros+counter_nosotras+counter_vosotros+counter_vosotras)*100)
#print("Number of male articles/pronouns", male_counter)
#print("Number of female articles/pronoungs", female_counter)
#print("Percentage of male articles/pronounds", (male_counter/(female_counter+male_counter)) * 100)


        
from Counterfactual import Counterfactual

import spacy
nlp = spacy.load("es_core_news_md")

from transformers import AutoTokenizer, BertModel
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
transformer = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

article_diff = ["él", "la"]
#articles = ['él', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas']
articles = ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'estas', 'estes', 'estos', 'aquello', 'aquella', 'aquellos', 'aquellas']
noun_diff = ["maestro", "maestra"]
hardcoded = {"nosotros": "nosotras"}
mf = Counterfactual(spacy=nlp, transformer=transformer, tokenizer=tokenizer,
                    noun_diff=noun_diff, article_diff=article_diff, articles=articles, hardcoded=hardcoded)

new_text = temp[0:100000]
female_total = counter_ella+counter_ellas+counter_vosotras+counter_nosotras
male_total = counter_el+counter_ellos+counter_nosotros+counter_vosotros
counter = 0
male_articles = ['el', 'él', 'los', 'un', 'unos', 'estos', 'aquello', 'aquellos']
indexes = []
for line in new_text:   
    counter += 1
    if counter % 1000 == 0: print(counter)
    if male_total > female_total*1.1:
        if " ellos " in line or " él " in line or " nosotros " in line or " vosotros " in line or "Nosotros" in line:
        #if any(word in line for word in male_articles):
        #print(line)
        #print(mf.flip_sentence(line))
            print(line)
            new_text.append(mf.flip_sentence(line))
            print(mf.flip_sentence(line))
            female_total+= 1
            #male_total -= 1
            indexes.append(new_text.index(line))
            #print(line)
    else:
        break
        #temp.append(line)

#new_text_array = np.asarray(new_text)
#np.save('fine_tune_data_new_correct.npy', new_text_array)
#new_text = np.load('fine_tune_data_new_correct.npy')
#print(new_text)

#np.save('counter_indexes_correct.npy', np.asarray(indexes))

print(len(new_text))
counter_ellos = 0
counter_ellas = 0
counter_el = 0
counter_ella = 0
counter_nosotros = 0
counter_nosotras = 0
counter_vosotros = 0
counter_vosotras = 0
male_counter = 0
female_counter = 0
for line in new_text:
    if " ellos " in line or "Ellos" in line: counter_ellos += 1 #print(line.replace('ellos', 'ellas'))
    if " ellas " in line or "Ellas" in line: counter_ellas += 1
    if " él " in line or "Él" in line: counter_el += 1
    if " ella " in line or "Ella" in line: counter_ella +=1
    if " nosotros " in line or "Nosotros" in line: counter_nosotros +=1
    if " nosotras " in line or "Nosotras" in line: counter_nosotras +=1
    if " vosotros " in line or "Vosotros" in line: counter_vosotros +=1
    if " vosotras " in line or "Nosotras" in line: counter_vosotras +=1
    #if any([word in line for word in male_articles]): male_counter += 1
    #if any([word in line for word in female_articles]): female_counter += 1

print("Number of ellos occurences: ", counter_ellos)
print("Number of ellas occurences: ", counter_ellas)
print("Number of el occurences: ", counter_el)
print("Number of ella occurences: ", counter_ella)
print("Number of nosotros occurences: ", counter_nosotros)
print("Number of nosotras occurences: ", counter_nosotras)
print("Number of vosotros occurences: ", counter_vosotros)
print("Number of vosotras occurences: ", counter_vosotras)
print("Percentage of male pronouns: ", (counter_el+counter_ellos+counter_nosotros+counter_vosotros)/(counter_el+counter_ella+counter_ellas+counter_ellos+counter_nosotros+counter_nosotras+counter_vosotros+counter_vosotras)*100)


#male_counter = 0
#female_counter = 0

#for line in temp[0:1000]:
    #if " ellos " in line or "Ellos" in line: counter_ellos += 1 #print(line.replace('ellos', 'ellas'))
    #if " ellas " in line or "Ellas" in line: counter_ellas += 1
    #if " él " in line or "Él" in line: counter_el += 1
    #if " ella " in line or "Ella" in line: counter_ella +=1
    #if any([word in line for word in male_articles]): male_counter += 1
    #if any([word in line for word in female_articles]): female_counter += 1

#print("Number of ellos occurences: ", counter_ellos)
#print("Number of ellas occurences: ", counter_ellas)
#print("Number of el occurences: ", counter_el)
#print("Number of ella occurences: ", counter_ella)
#print("Percentage of male pronouns: ", (counter_el+counter_ellos)/(counter_el+counter_ella+counter_ellas+counter_ellos)*100)
#print("Number of male articles/pronouns", male_counter)
#print("Number of female articles/pronoungs", female_counter)
#print("Percentage of male articles/pronounds", (male_counter/(female_counter+male_counter)) * 100)

