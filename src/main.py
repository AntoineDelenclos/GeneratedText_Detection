from src.dataset import scrapingFromText, createDataset
from src.tokenization import tokenizeByCharacter


def main():
    SENTENCES_NUMBER = 2000
    TRAINING_PERCENT = 0.8
    WATERMARK_PERCENT = 0.5
    EMBEDDING_DIM = 50
    PADDING = 'post'

    sentences = scrapingFromText("C:/Users/antoi/PycharmProjects/GeneratedText_Detection/data/emile.txt", SENTENCES_NUMBER)
    trainingDataset, validationDataset, testDataset, trainingWatermarkBooleans, validationWatermarkBooleans, testWatermarkBooleans = createDataset(sentences, TRAINING_PERCENT, WATERMARK_PERCENT)
    print(trainingDataset)

    paddedCharSequences, embeddingMatrix = tokenizeByCharacter(trainingDataset, EMBEDDING_DIM, PADDING)
    print(embeddingMatrix[0])

if __name__ == "__main__":
    main()