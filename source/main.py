from multiprocessing import cpu_count
import pandas as pd
import time
import os

from source.data_processing.utils.utils import save_tree
from source.data_processing.utils.utils import load_data
from source.data_processing.utils.utils import format_metadata

from source.data_processing.tree.tree import build_tree
from source.data_processing.tree.tree import search_models_tree
from source.data_processing.tree.tree import build_dataframe

from source.data_processing.input.input_processing import InputProcessing

from source.data_processing.train_scheduler.train_scheduler import TrainScheduler

from source.data_processing.models.scale_models import scaling_models

from source.data_processing.data_augmentation import data_augmentation


def set_data(data_path, chunksize, usecols):
    print('Reading data...')
    data_generator = load_data(data_path=data_path, chunksize=chunksize, usecols=usecols)

    print('Preprocessing data...')
    input_processing = InputProcessing(data_generator)
    data_generator = input_processing.process_input_generator(usecols)
    return data_generator


def final_tree(dataset_generator, usecols):
    dataset = pd.DataFrame(dataset_generator, columns=usecols.values())
    dataset.index = range(len(dataset))

    print('Building tree...')
    starting = set(dataset[usecols['output']].apply(lambda x: x.split('/')[0]))

    tree_list = []
    for category in starting:
        tree = build_tree(category=category, dataset=dataset, column_output=usecols['output'])
        tree_list.append(tree)
        save_tree(category, tree)

    print('Building tree metadata...')
    metadata = search_models_tree(tree_list, starting)
    metadata = format_metadata(metadata)

    print("Metadata's number: ", len(metadata))
    return metadata, tree_list, starting, dataset


def build_datasets(tree_list, starting, dataset, queue_models, column_output):
    print('Building training datasets')
    datasets_metrics = build_dataframe(tree_list=tree_list, category_list=starting,
                                       dataframe=dataset, queue_models=queue_models,
                                       column_output=column_output)
    datasets_metrics = pd.DataFrame(datasets_metrics, columns=[''])
    datasets_metrics.to_csv('../data/metrics/dataset_len.csv')


def augment_data():
    print('Doing data augmentation')
    models_list = os.listdir('../data/dataset/training/')
    dataframe = pd.DataFrame(columns=['model', 'len'])
    for model in models_list:
        df = pd.read_csv('../data/dataset/training/' + model)
        dataframe = dataframe.append([{'model': model, 'len': len(df)}])
    dataframe.to_csv('../data/metrics/df_len.csv')
    data_augmentation(dataframe)


def train_models(metadata, usecols, epochs):
    print('Training models...')
    n_processes = cpu_count() - 1
    train_container = TrainScheduler(n_processes=n_processes)
    train_container.initialize_queue(metadata)
    train_container.initialize_process(usecols=usecols, epochs=epochs)
    return train_container.success, train_container.fail


def run_main(path, usecols, chunksize):
    start = time.time()
    dataset_generator = set_data(path, chunksize, usecols)
    metadata, tree_list, starting, dataset = final_tree(dataset_generator, usecols)
    build_datasets(tree_list, starting, dataset, queue_models=metadata,
                   column_output=usecols['output'])
    augment_data()
    del dataset
    models_name, fail = train_models(metadata=metadata, usecols=usecols, epochs=30)
    print("Failed training {0} models".format(fail))
    print(fail)
    scaling_models(model_dir_path='../data/trained_models/models/')
    end = time.time()
    print('Took {0:.2f} min to execute'.format((end - start) / 60))


if __name__ == '__main__':
    run_main(path='../data/dataset/processed_data/tabular_data.csv',
             usecols={'input': 'query_string', 'output': 'output'},
             chunksize=100000)
