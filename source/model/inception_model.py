from keras import Input
from keras.models import Model
from keras import layers
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'


def tf_no_warning():
    """
    Make Tensorflow less verbose
    """
    try:

        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except ImportError:
        pass


class B2WInception(object):
    def __init__(self, n_classes):
        self._history = None
        self.__embedding_model = None
        self.__network_model = None
        self.n_classes = n_classes
        self.__history = None

        tf_no_warning()

    def __build_embedding(self):
        # Creating a model related only to create the char embedding
        # Model 1
        input_data = Input(batch_shape=(None, 42))
        x = layers.Embedding(42, 30)(input_data)
        self.__embedding_model = Model(input_data, x)

    def __build_estimator(self):
        input_model_1 = Input(batch_shape=(None, 42))

        conv1d_layer = self.__embedding_model(input_model_1)

        conv1d_one_1 = layers.Conv1D(filters=30, kernel_size=1, strides=1, activation='relu')
        conv1d_two_1 = layers.Conv1D(filters=30, kernel_size=2, strides=1, activation='relu')

        conv_1 = conv1d_one_1(conv1d_layer)
        conv_2 = conv1d_two_1(conv1d_layer)

        concatenate_1 = layers.concatenate([conv_1, conv_2], axis=1)

        x = layers.MaxPooling1D()(concatenate_1)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dense(1024, activation='relu')(x)

        output = layers.Dense(self.n_classes, activation='softmax')(x)

        self.__network_model = Model(input_model_1, output)

    def build_model(self):
        self.__build_embedding()
        self.__build_estimator()

        optimizer = optimizers.Adam()

        self.__network_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['acc'])

    def fit_model(self, training_generator, validation_generator, epochs, model_path, workers=0):
        # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=10)
        best_model = ModelCheckpoint(model_path + '.h5', monitor='val_acc', mode='max', verbose=True,
                                     save_best_only=True)
        callbacks = [best_model]

        self.__history = self.__network_model.fit_generator(generator=training_generator, epochs=epochs,
                                                            use_multiprocessing=True,
                                                            validation_data=validation_generator,
                                                            verbose=True, callbacks=callbacks, workers=workers)
        return self.__network_model

    def evaluate_model(self, testing_generator):
        return self.__network_model.evaluate_generator(testing_generator, steps=None, max_queue_size=10, workers=0,
                                                       use_multiprocessing=True, verbose=True)[1]

