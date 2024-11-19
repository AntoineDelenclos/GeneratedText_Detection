import numpy as np
import pandas as pd
import re

def scrapingFromText(path, sentencesNumber):
    with open(path,'r',encoding='utf-8') as f:
        content = f.read()

    content.replace('\n', '')
    sentences = re.split(r'(?<=[.?!])\s+',content) #This pattern will split on ".", "?", "!", "\n", followed by any whitespace

    non_empty_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    dataset_sentences = non_empty_sentences[:sentencesNumber]
    i = 0
    for sentence in dataset_sentences:
        if '\n' in sentence:
            dataset_sentences[i] = sentence.replace('\n', ' ')
        i += 1

    print(dataset_sentences)
    with open('data/dataset_sentences.txt', 'w', encoding='utf-8') as output_file:
        for sentence in dataset_sentences:
            output_file.write(sentence + "\n")

    return 0
