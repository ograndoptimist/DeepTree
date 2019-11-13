import multiprocessing as mp

from source.model.inception_model import B2WInception
from source.data_processing.utils.utils import format_path_name, classes_lookup, build_dataset, clean_dataset
from source.data_processing.data_generator import DataGenerator


class MultiprocessingTrain(object):
    def __init__(self, epochs):
        self.__epochs = epochs
        self.__mp_queue = mp.Queue()
        self.__processes = None

    def __build_processes(self, metadata_container, usecols):
        # Setup a list of processes that we want to run
        self.__processes = [mp.Process(target=self.__get_task,
                                       args=(
                                           B2WInception(n_classes=len(metadata['children'])),
                                           DataGenerator(path=format_path_name(root_dir='../data/dataset/training/',
                                                                               model_name=metadata['model_name'],
                                                                               mode='_train.csv'),
                                                         batch_size=16,
                                                         usecols=usecols,
                                                         n_classes=len(metadata['children']),
                                                         lookup_path=classes_lookup(metadata), mode='train'),
                                           DataGenerator(path=format_path_name(root_dir='../data/dataset/training/',
                                                                               model_name=metadata['model_name'],
                                                                               mode='_test.csv'),
                                                         batch_size=16,
                                                         usecols=usecols,
                                                         n_classes=len(metadata['children']),
                                                         lookup_path=classes_lookup(metadata), mode='test'),
                                           DataGenerator(path=format_path_name(root_dir='../data/dataset/training/',
                                                                               model_name=metadata['model_name'],
                                                                               mode='_validation.csv',
                                                                               ),
                                                         batch_size=16,
                                                         usecols=usecols,
                                                         n_classes=len(metadata['children']),
                                                         lookup_path=classes_lookup(metadata), mode='validation'),
                                           '../data/trained_models/models/' + str(metadata['parent_category'])
                                           + '_' + str(metadata['root']) + '_level_' + str(metadata['level']), metadata)
                                       )
                            for index, metadata in enumerate(metadata_container) if metadata['result'] is True]

    def __try_task(self, task, train_generator, test_generator, validation_generator, path, metadata):
        try:
            if metadata['result'] is False:
                return metadata
            result = \
                task.fit_model(train_generator, validation_generator, epochs=self.__epochs, model_path=path,
                               workers=0).evaluate_generator(test_generator)[1]
            return {**metadata, 'result': result}
        except ValueError:
            return {**metadata, 'result': False}

    def __get_task(self, task, train_generator, test_generator, validation_generator, path, metadata):
        """
            Define the tasks that will be inserted in the mp.Queue().
        """
        task.build_model()
        self.__mp_queue.put(
            self.__try_task(task, train_generator, test_generator, validation_generator,
                            path, metadata)
        )

    @staticmethod
    def __preprocess_dataset(metadata, usecols):
        model_name = metadata['root'] + '_level_' + str(metadata['level'])
        try:
            path_dict = build_dataset(path=format_path_name(root_dir='../data/dataset/training/',
                                                            model_name=model_name, mode='.csv'),
                                      model_name=model_name, usecols=usecols)
            [clean_dataset(path_dict[key], columns_input=usecols['input']) for key in path_dict.keys()]
            return {**metadata, 'model_name': model_name, 'result': True}
        except FileNotFoundError:
            return {**metadata, 'model_name': model_name, 'result': False}

    def train_models(self, metadata_container, usecols):
        metadata_container = [MultiprocessingTrain.__preprocess_dataset(metadata, usecols)
                              for metadata in metadata_container]
        self.__build_processes(metadata_container, usecols)
        self.__run_processes()
        self.__exit_processes()
        results = self.__get_processes()
        return results

    def __run_processes(self):
        # Run processes
        for process in self.__processes:
            process.start()

    def __exit_processes(self):
        # Exit the completed processes
        for process in self.__processes:
            process.join()

    def __get_processes(self):
        # Get processes results from the output queue
        return [self.__mp_queue.get() for process in self.__processes]
