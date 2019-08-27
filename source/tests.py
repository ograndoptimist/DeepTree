import pandas as pd


from source.data_processing.input.input_processing import InputProcessing


def load_data(data_path, chunksize):
    """
        Returns a generator from the dataset.
    """
    data_reference = pd.read_csv(data_path, chunksize=chunksize, usecols=['query_string', 'output'])
    return data_reference


def save_data(data_reference, path):
    with open(path, "w") as file:
        pd.DataFrame(data_reference, columns=['query_string', 'output']).to_csv(path, index=None)


if __name__ == '__main__':
    data_generator = load_data(data_path='../data/dataset/processed_data/tabular_data.csv',
                               chunksize=100000)
    input_processing = InputProcessing(data_generator)
    data = input_processing.process_input()
    save_data(data, path='../data/dataset/processed_data/input.csv')
