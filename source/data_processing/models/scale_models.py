import os

from keras.models import load_model
import keras.backend as K


def scaling_models(model_dir_path):
    print('Scaling models')
    models_list = os.listdir(model_dir_path)
    for model_name in models_list:
        model_path = model_dir_path + model_name
        model = load_model(model_path)

        model_name = '_'.join(model_name.split('_')[1:]).replace('.h5', '')
        print('\t Scaling model {0}'.format(model_name))

        final_path_weight = '../data/trained_models/weights/' + model_name + '.h5'
        model.save_weights(final_path_weight)

        final_path_architecture = '../data/trained_models/architectures/' + model_name + '.json'
        arch = model.to_json()
        with open(final_path_architecture, 'w') as file:
            file.write(arch)

        K.clear_session()
        del model
