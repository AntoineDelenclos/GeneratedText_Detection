import numpy as np
from src.dataset import scrapingFromText, createDataset
from src.model import watermarkClassifier
from src.preparation import tokenizeByCharacter, tokenizeByWord, embeddingToken, tokenizeByWordAndEmbedding


def main():
    SENTENCES_NUMBER = 30
    TRAINING_PERCENT = 0.8
    WATERMARK_PERCENT = 0.5
    NEURONS = 16
    CONVOLUTION_NUMBER = 3
    DROPOUT = 0.3
    POOL_SIZE = 2 # 2 class classifier
    EPOCHS = 20
    BATCH_SIZE = 16

    sentences = scrapingFromText("/home/antoine/PycharmProjects/GeneratedText_Detection/data/emile.txt", SENTENCES_NUMBER)
    trainingDataset, validationDataset, testDataset, trainingWatermarkedBooleans, validationWatermarkedBooleans, testWatermarkedBooleans = createDataset(sentences, TRAINING_PERCENT, WATERMARK_PERCENT)


    trainingTokens = np.array(tokenizeByWordAndEmbedding(trainingDataset))
    validationTokens = np.array(tokenizeByWordAndEmbedding(validationDataset))
    testTokens = np.array(tokenizeByWordAndEmbedding(testDataset))

    trainingWatermarkedBooleans = np.array(trainingWatermarkedBooleans)
    validationWatermarkedBooleans = np.array(validationWatermarkedBooleans)
    testWatermarkedBooleans = np.array(testWatermarkedBooleans)

    model = watermarkClassifier(trainingTokens.shape[1], NEURONS, CONVOLUTION_NUMBER, DROPOUT, POOL_SIZE)

    history = model.fit(
        trainingTokens, trainingWatermarkedBooleans,
        validation_data = (validationTokens, validationWatermarkedBooleans),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    print(model.evaluate(testTokens, testWatermarkedBooleans))

if __name__ == "__main__":
    main()