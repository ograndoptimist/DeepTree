from nltk.corpus import stopwords
from unidecode import unidecode
from keras.preprocessing.text import text_to_word_sequence
from string import punctuation


class InputProcessing(object):
    """
        A class only responsible to clean input text dataset.
    """

    def __init__(self, data_generator=None):
        if data_generator is not None:
            self.data_reference = data_generator
        self.stop_words = stopwords.words('portuguese')

    @staticmethod
    def __remove_punctuation(raw_text):
        raw_text = raw_text.translate(str.maketrans('', '', punctuation))
        return raw_text

    @staticmethod
    def __to_lower(raw_text):
        raw_text = str(raw_text).lower()
        return raw_text

    @staticmethod
    def __remove_accent(raw_text):
        raw_text = unidecode(raw_text)
        return raw_text

    def __remove_stopwords(self, list_text):
        list_text = [word for word in list_text if word not in self.stop_words]
        return list_text

    @staticmethod
    def __split_text(raw_text):
        raw_text = text_to_word_sequence(raw_text)
        return raw_text

    def process_input(self, data):
        data = InputProcessing.__remove_punctuation(data)
        data = InputProcessing.__to_lower(data)
        data = InputProcessing.__remove_accent(data)
        data = InputProcessing.__split_text(data)
        data = self.__remove_stopwords(data)
        data = ' '.join(data)
        return data

    def process_input_generator(self, usecols):
        assert self.data_reference is not None
        for data_block in self.data_reference:
            for data, output in zip(data_block[usecols['input']], data_block[usecols['output']]):
                data = self.process_input(data)
                yield data, output
