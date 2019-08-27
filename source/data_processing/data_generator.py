import keras
from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd

from source.data_processing.utils.utils import read_dic
from source.data_processing.utils.utils import stack_array

from source.data_processing.data_vectorizer import Vectorizer


class DataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size, n_classes, lookup_path, usecols, data_len=256, mode='train'):
        self.batch_size = batch_size
        self.mode = mode
        self.n_classes = n_classes
        self.path = path
        self.__usecols = usecols
        self.__data_len = data_len
        self.__data_object = self.__read_data()

        self.__vectorizer = Vectorizer(tokenizer=Tokenizer(char_level=True),
                                       tokens_char=read_dic('../data/tokens/tokens_char.txt'),
                                       class_labels=read_dic(lookup_path))

    def __read_data(self):
        return pd.read_csv(self.path, chunksize=self.batch_size, usecols=self.__usecols.values())

    def __len__(self):
        """
            Denotes the number of batches per epoch.
        """
        return int(np.floor(self.__data_len / self.batch_size))

    def __reset_dataset(self):
        self.__data_object = self.__read_data()

    def __generate_data(self, input_data=None, output_data=None, repass=None):
        if input_data is None:
            check_intern = 0
        else:
            check_intern = repass

        for data_block in self.__data_object:
            for cont, x, y in zip(range(data_block.shape[0]),
                                  data_block[self.__usecols['input']],
                                  data_block[self.__usecols['output']]):
                item_input = self.__vectorizer.vectorize_input(x)
                item_output = self.__vectorizer.vectorize_output(y)

                if cont == 0 and input_data is None:
                    input_data = np.copy(item_input)
                    output_data = np.copy(item_output)
                else:
                    input_data = stack_array(arr1=input_data, arr2=item_input, dim=item_input.shape[0])
                    output_data = stack_array(arr1=output_data, arr2=item_output, dim=item_output.shape[0])

                check_intern += 1

                if check_intern == self.batch_size:
                    break
            break

        if check_intern < self.batch_size:
            self.__reset_dataset()
            return self.__generate_data(input_data=input_data, output_data=output_data, repass=check_intern)

        return input_data, output_data

    def __getitem__(self, index):
        """
            Generate one batch of dataset.
        """
        X, y = self.__generate_data()
        return X, y
