from copy import copy

from source.data_processing.train_scheduler.data_container import DataContainer
from source.data_processing.train_scheduler.multiprocessing_train import MultiprocessingTrain


class TrainScheduler(object):
    """
        Responsible to manage the training of multiples models.
        You can specify the number of models to train in parallel.
    """
    def __init__(self, n_processes):
        self.__n_processes = n_processes
        self.__main_stack = DataContainer()
        self.__aux_stack = DataContainer()
        self.__data_flow = None
        self.__success_stack = []
        self.__fail_stack = []

    @property
    def success(self):
        return self.__success_stack

    @property
    def fail(self):
        return self.__fail_stack

    def initialize_queue(self, tasks):
        [self.__main_stack.add(data) for data in tasks]
        self.__data_flow = copy(self.__main_stack)

    def __data_add(self):
        data_flow_length = len(self.__data_flow)
        for task in range(data_flow_length):
            if task == self.__n_processes:
                break
            self.__aux_stack.appendright(self.__data_flow.popleft())

    def __data_remove(self):
        aux_stack_length = len(self.__aux_stack)
        for task in range(aux_stack_length):
            if task == self.__n_processes:
                break
            self.__aux_stack.popleft()

    @staticmethod
    def __start_train(metadata_container, usecols, epochs):
        mp_train = MultiprocessingTrain(epochs)
        return mp_train.train_models(metadata_container, usecols)

    def __pick_data(self):
        metadata_container = [metadata['metadata']['metadata'] for metadata in self.__aux_stack]
        self.__data_remove()
        return metadata_container

    def initialize_process(self, usecols, epochs):
        self.__data_add()
        while not self.__aux_stack.is_empty():
            metadata_container = self.__pick_data()
            results = TrainScheduler.__start_train(metadata_container, usecols, epochs)
            self.__send_stack(results)
            self.__data_add()

    def __send_stack(self, results):
        for model in results:
            if model['result'] is not None:
                self.__success_stack.append(model['model_name'])
            else:
                self.__fail_stack.append(model['model_name'])
