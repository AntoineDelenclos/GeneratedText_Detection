import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences


def tokenizeByCharacter(dataset, embeddingDim, padding):
    charTokenizer = Tokenizer(char_level=True, oov_token='<OOV>') #oov :
    charTokenizer.fit_on_texts(dataset)

    charSequences = charTokenizer.texts_to_sequences(dataset)
    maxLength = max(len(sequence) for sequence in charSequences) #Maybe try to do the max of all the sequences (training, validation and test sequences)

    paddedCharSequences = pad_sequences(charSequences, maxlen=maxLength, padding=padding) #Need to normalize the sequences as they are not all the same length

    vocabSize = len(charTokenizer.word_index) + 1
    embeddingMatrix = np.random.rand(vocabSize, embeddingDim)

    return paddedCharSequences, embeddingMatrix

