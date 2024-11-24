import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input


def watermarkClassifier(inputLength, neuronsNumber, convolutionNumber, dropout, poolSize):
    model = Sequential()
    print(convolutionNumber, poolSize)
    # Convolution 1D pour capturer les motifs dans les vecteurs
    model.add(Input(shape=(inputLength, ))) #It will find the corresponding embedding
    model.add(Dense(units=neuronsNumber, activation='relu'))

    # Couche dense pour la classification
    #model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid pour une classification binaire

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
