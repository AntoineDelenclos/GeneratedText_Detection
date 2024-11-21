import random
import re
import numpy as np


#This function will take a text file and return another file where each line contains a single sentence.
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

    # print(dataset_sentences)
    # with open("C:/Users/antoi/PycharmProjects/GeneratedText_Detection/data/dataset_sentences.txt", 'w', encoding='utf-8') as output_file:
    #     for sentence in dataset_sentences:
    #         output_file.write(sentence + "\n")

    return dataset_sentences

def watermarkSentence(sentence): #Use of the EasyMark watermark (leverages whitespaces)
    return sentence.replace(chr(0x0020), chr(0x2004))

def sentenceIsWatermarked(sentence):
    return chr(0x2004) in sentence

#This function will create a dataset that will contain : training, validation and test data
def createDataset(sentences, trainingPercent, watermarkPercent):
    trainingDataset, validationDataset, testDataset = [], [], []
    trainingWatermarkBooleans, validationWatermarkBooleans, testWatermarkBooleans = [], [], []
    validationPercent = (1 - trainingPercent)/2
    testPercent = (1 - trainingPercent)/2

    numberOfSentences = len(sentences)
    countTraining = int(numberOfSentences * trainingPercent)
    countValidation = int(numberOfSentences * validationPercent) + countTraining
    countTest = int(numberOfSentences * testPercent) + countValidation

    print(f"training : {countTraining}, validation : {countValidation}, test : {countTest}")

    count = 0
    for sentence in sentences:
        if count < countTraining:
            print("TRAIN")
            trainingDataset.append(sentence)
            if random.randint(0, 1) <= watermarkPercent:
                newSentence = watermarkSentence(sentence)
                trainingDataset.append(newSentence)

        elif count < countValidation:
            print("VAL")
            validationDataset.append(sentence)
            if random.randint(0, 1) <= watermarkPercent:
                newSentence = watermarkSentence(sentence)
                validationDataset.append(newSentence)

        elif count < countTest:
            print("TEST")
            testDataset.append(sentence)
            if random.randint(0,1) <= watermarkPercent:
                newSentence = watermarkSentence(sentence)
                testDataset.append(newSentence)

        count += 1

    for i in range(len(trainingDataset)):
        if sentenceIsWatermarked(trainingDataset[i]):
            trainingWatermarkBooleans.append(True)
        else:
            trainingWatermarkBooleans.append(False)
    for i in range(len(validationDataset)):
        if sentenceIsWatermarked(validationDataset[i]):
            validationWatermarkBooleans.append(True)
        else:
            validationWatermarkBooleans.append(False)
    for i in range(len(testDataset)):
        if sentenceIsWatermarked(testDataset[i]):
            testWatermarkBooleans.append(True)
        else:
            testWatermarkBooleans.append(False)

    return trainingDataset, validationDataset, testDataset, np.array(trainingWatermarkBooleans), np.array(validationWatermarkBooleans), np.array(testWatermarkBooleans)
