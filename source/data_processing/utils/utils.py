import pandas as pd
from sklearn.model_selection import train_test_split
from ast import literal_eval
import pickle
import tensorflow as tf

from numpy import nan
from numpy import concatenate


def format_path_name(root_dir, model_name, mode):
    return "{0}{1}{2}".format(root_dir, model_name, mode)


def save_class_lookup(classes_lookup, path):
    with open(path, 'w') as file:
        file.write(str(classes_lookup))


def build_classes_lookup(metadata):
    classes_lookup = dict()
    for index, class_ in enumerate(metadata['children']):
        classes_lookup[class_] = index
    return classes_lookup


def classes_lookup(metadata):
    classes_lookup = build_classes_lookup(metadata)
    path = format_path_name(root_dir='../data/labels/', model_name=metadata['model_name'], mode='.txt')
    save_class_lookup(classes_lookup, path)
    return path


def split_dataset(dataframe):
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    X_train, X_test_, y_train, y_test_ = train_test_split(X, y, random_state=42, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test_, y_test_, random_state=42, test_size=0.33)

    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1), pd.concat(
        [X_validation, y_validation], axis=1)


def save_dataframe(dataframe, path, usecols):
    try:
        file = open(path)
        file.close()
        dataframe.to_csv(path, mode='a', columns=usecols, index=False, header=None)
    except FileNotFoundError:
        dataframe.to_csv(path, columns=usecols, index=False)


def build_dataset(path, model_name, usecols):
    try:
        train_path = format_path_name(root_dir='../data/dataset/training/', model_name=model_name, mode='_train.csv')
        test_path = format_path_name(root_dir='../data/dataset/training/', model_name=model_name, mode='_test.csv')
        validation_path = format_path_name(root_dir='../data/dataset/training/', model_name=model_name,
                                           mode='_validation.csv')

        dataset = pd.read_csv(path, usecols=usecols.values())
        train, test, validation = split_dataset(dataset)

        train.to_csv(train_path)
        test.to_csv(test_path)
        validation.to_csv(validation_path)
        return {'train': train_path, 'test': test_path, 'validation': validation_path}
    except TypeError:
        print('usecols: ', usecols)


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def drop_row(dataset, column):
    df = dataset.copy()

    for index in range(len(df)):
        if represents_int(df.loc[index, column]):
            df.loc[index, column] = nan

    df = df.dropna()
    df.index = range(len(df))
    return df


def clean_dataset(path, columns_input):
    dataset = pd.read_csv(path)
    dataset = drop_row(dataset, columns_input)
    dataset = drop_row(dataset, columns_input)
    dataset.to_csv(path)


def read_dic(path):
    with open(path, 'r') as file:
        element = file.read()
        element = literal_eval(element)
    return element


def configure_paths(first_inference, hierarchy_level):
    model_name = format_path_name(first_inference, '_level_', hierarchy_level)
    model_weight_path = format_path_name(root_dir='../../data/trained_models/weights/',
                                         model_name=model_name, mode='.h5')
    class_path = format_path_name(root_dir='../../data/labels/', model_name=model_name, mode='.txt')
    path_model_architecture = format_path_name(root_dir='../../data/trained_models/architectures/',
                                               model_name=model_name, mode='.json')
    return model_weight_path, class_path, path_model_architecture


def invert_dict(dic):
    new_dict = dict()
    for k, v in zip(dic.keys(), dic.values()):
        new_dict[v] = k
    return new_dict


def save_tree(category, tree):
    with open('../data/trees/' + category + '.pickle', 'wb') as file:
        pickle.dump(tree, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_tree(path_tree):
    with open(path_tree, 'rb') as file:
        tree = pickle.load(file)
    return tree


def have_children(tree):
    return len(tree.children(tree.root)) > 1


def stack_array(arr1, arr2, dim, axis=0):
    arr1 = arr1.reshape(arr1.shape[0], dim) if arr1.ndim > 1 else arr1.reshape(arr2.ndim, dim)
    arr2 = arr2.reshape(arr2.ndim, dim)
    return concatenate((arr1, arr2), axis=axis)


def load_data(data_path, chunksize, usecols):
    """
        Returns a generator from the dataset.
    """
    data_reference = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols.values())
    return data_reference


def format_metadata(vec_metadata):
    metadata_formatted = []
    for vec in vec_metadata:
        for metadata in vec:
            metadata_formatted.append(metadata)
    return metadata_formatted
