import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import spacy



def tokenizeByCharacter(dataset, embeddingDim, padding):
    charTokenizer = Tokenizer(char_level=True, oov_token='<OOV>') #oov :
    charTokenizer.fit_on_texts(dataset)

    charSequences = charTokenizer.texts_to_sequences(dataset)
    maxLength = max(len(sequence) for sequence in charSequences) #Maybe try to do the max of all the sequences (training, validation and test sequences)

    paddedCharSequences = pad_sequences(charSequences, maxlen=maxLength, padding=padding) #Need to normalize the sequences as they are not all the same length

    vocabSize = len(charTokenizer.word_index) + 1
    embeddingMatrix = np.random.rand(vocabSize, embeddingDim)

    return paddedCharSequences, embeddingMatrix

def tokenizeByWordAndEmbedding(dataset): #Do tokenization and embedding in the same time is quicker
    embeddedTokens = []
    nlp = spacy.load("en_core_web_sm")
    for sentence in dataset:
        doc = nlp(sentence)
        embeddedTokens.append(np.mean([X.vector for X in doc], axis=0)) #Tokenized and embedded
    return embeddedTokens

def tokenizeByWord(dataset):
    tokenizedDataset = []
    nlp = spacy.load("en_core_web_sm")
    for sentence in dataset:
        doc = nlp(sentence)
        tokenizedDataset.append([X.text for X in doc])
    return tokenizedDataset

def embeddingToken(tokenizedDataset, embeddingDim):
    nlp = spacy.load("en_core_web_sm")
    embeddedTokens = []
    for tokenizedSentence in tokenizedDataset:
        doc = nlp(tokenizedSentence)
        embeddedTokens.append([X.vector for X in doc])