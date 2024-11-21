from src.dataset import scrapingFromText, createDataset
from src.model import watermarkClassifier
from src.preparation import tokenizeByCharacter


def main():
    SENTENCES_NUMBER = 2000
    TRAINING_PERCENT = 0.8
    WATERMARK_PERCENT = 0.5
    EMBEDDING_DIM = 50
    PADDING = 'post'
    EPOCHS = 0
    BATCH_SIZE = 0

    sentences = scrapingFromText("C:/Users/antoi/PycharmProjects/GeneratedText_Detection/data/emile.txt", SENTENCES_NUMBER)
    trainingDataset, validationDataset, testDataset, trainingWatermarkBooleans, validationWatermarkBooleans, testWatermarkBooleans = createDataset(sentences, TRAINING_PERCENT, WATERMARK_PERCENT)
    print(trainingDataset)

    trainingPaddedCharSequences, trainingEmbeddingMatrix = tokenizeByCharacter(trainingDataset, EMBEDDING_DIM, PADDING)
    validationPaddedCharSequences, validationEmbeddingMatrix = tokenizeByCharacter(validationDataset, EMBEDDING_DIM, PADDING)
    testPaddedCharSequences, testEmbeddingMatrix = tokenizeByCharacter(testDataset, EMBEDDING_DIM, PADDING)

    model = watermarkClassifier(50, EMBEDDING_DIM)

    history = model.fit(
        trainingPaddedCharSequences, trainingWatermarkBooleans,
        validation_data = (validationPaddedCharSequences, validationWatermarkBooleans),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    prediction = model.predict(testPaddedCharSequences)

    print(testPaddedCharSequences)

if __name__ == "__main__":
    main()