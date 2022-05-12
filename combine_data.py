import pandas as pd
import numpy as np
import re
import numpy as np

with open('./data/europarl-v7.es-en.en', encoding="utf-8") as file:
        # Read the line from file, strip leading and trailing whitespace,
        # prepend the start-text and append the end-text.
        texts_en = ["" + line.strip() + "" for line in file]

indexes = np.load("counter_indexes.npy")
spanish_translations = np.load("fine_tune_data_new.npy")

temp = texts_en
counter_he = 0
counter_she = 0
counter_him = 0
counter_her = 0
male_counter = 0
female_counter = 0
male_pro = ['he', 'He', 'him', 'Him', 'his']
female_pro = [' she ', 'She', ' her ', 'Her']
change_dict = {' he ': ' she ', 'He': 'She', ' him ':' her ', 'Him':'Her', ' his ':' her '}

new_text = texts_en[0:100000]
counter = 0
for idx in indexes:
    #for word in temp[idx].split():
    counter += 1
    for words in change_dict:
        if re.search(r"\b" + re.escape(words) + r"\b", temp[idx]):
        #print(word)
        #if words in change_dict.keys():
            #print(words)
            print("HERE\n")
            #print(temp[idx])
            temp[idx] = temp[idx].replace(words, change_dict[words])
            #print(temp[idx])
            #print(idx)
    new_text.append(temp[idx])
    #print(temp[idx])
    #print(counter)

print(len(new_text))
#np.save("fine_tune_data_en.npy", new_text)
print(len(spanish_translations))



# COMBINE DATASETS HERE
new_text = np.load("fine_tune_data_en.npy")
full_data = []
for i in range(0,len(new_text)):
    full_data.append({'id': i, 'translation': {'en': new_text[i], 'es':spanish_translations[i]}})
print(full_data[0])

#np.save("fine_tune_full_data.npy", np.asarray(full_data))



#print(np.where(indexes==26482))
#print(spanish_translations[100482])


    #print(temp[idx])
    #print(temp[idx].replace(word, change_dict[word]))
    #new_text.append(temp[idx]

# what if do a dict correspondance, so like {he: she,  

#for line in temp[0:150]:
    #if any(word in line for word in male_pro): 
    #    print(line)

    #    print(line.replace(word, change_dict[word]) for word in male_pro if word in line )
    #    male_counter += 1
        
    #if any(word in line for word in female_pro): female_counter += 1

    #if " he " in line or "He" in line: counter_he += 1 #print(line.replace('ellos', 'ellas'))
    #if " she " in line or "She" in line: counter_she += 1
    #if " him " in line or "Him" in line: 
        #counter_him += 1
        #print(line)
        #print(line.replace(" him ", " her "))
    #if " her " in line or "Her" in line: counter_her +=1

#print("Number of he occurences: ", counter_he)
#print("Number of she occurences: ", counter_she)
#print("Number of him occurences: ", counter_him)
#print("Number of her occurences: ", counter_her)
#print("Percentage of male pronouns: ", (counter_he+counter_him)/(counter_he+counter_she+counter_him+counter_her)*100)
#print("Percentage of male pronouns: ", (male_counter/female_counter*100))

