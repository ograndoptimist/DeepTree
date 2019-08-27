import numpy as np
from keras.utils import to_categorical
from unidecode import unidecode


class Vectorizer:
    def __init__(self, tokenizer, tokens_char, class_labels):
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(tokens_char)
        self.class_labels = class_labels

    def vectorize_input(self, data):
        data = unidecode(data)
        data = data.replace('_', '')
        data = data.replace("'", '')
        try:
            x = self.tokenizer.texts_to_sequences(data)
            arr = np.zeros((len(x), 1))
            for i, token in enumerate(x):
                arr[i] = token[0]
            arr = arr.reshape(1, -1)
            arr = np.pad(arr[0], (42 - len(arr[0]), 0), 'constant')
            return arr
        except IndexError:
            print(data)
            raise

    def vectorize_output(self, data):
        try:
            data = self.class_labels[data]
            data = to_categorical(data, num_classes=len(self.class_labels))
            return data
        except KeyError:
            raise
