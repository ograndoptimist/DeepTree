import numpy as np

from source.data_processing.input.input_processing import InputProcessing
from source.data_processing.utils.utils import read_dic


class InputProvider:
    """
        A class responsible to convert the raw text from input to
        the correct format towards the model.
    """
    def __init__(self):
        self._tokens = read_dic(path='../../data/tokens/tokens_char.txt')

    @staticmethod
    def __make_tensor(data):
        tensor_input = [0] * 42
        for item in range(len(data)):
            if item == 42:
                break
            tensor_input[-1 - item] = data[-1 - item]
        return np.array(tensor_input).reshape(1, 42)

    def process_data(self, data):
        input_processing = InputProcessing()
        data = list(input_processing.process_input(data))
        data = [self._tokens[character] for character in data]
        tensor_input = InputProvider.__make_tensor(data)
        return tensor_input

