from keras.models import model_from_json
from ast import literal_eval
import numpy as np


from .input_provider import InputProvider
from source.data_processing.utils.utils import invert_dict


class Canonizador:
    def __init__(self, path_model_weights, path_class, path_model_architecture):
        with open(path_model_architecture, 'r') as file:
            model_architecture = file.read()
            self.model = model_from_json(model_architecture)
            self.model.load_weights(path_model_weights)

        with open(path_class, 'r')as file:
            check = file.read()
            check = literal_eval(check)
            self._dict_class = invert_dict(check)

    def predict_model(self, input_data):
        data = InputProvider().process_data(input_data)
        return self._dict_class[np.argmax(self.model.predict(data))]

    def predict_describe_model(self, input_data):
        data = InputProvider().process_data(input_data)
        aux_array = np.array(self.model.predict(data))[0]
        details = sorted(aux_array, reverse=True)[:3]
        return {self._dict_class[np.where(aux_array == k)[0][0]]: "{:.2f}".format(k * 100) for k in details}