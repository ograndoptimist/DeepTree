import pandas as pd


def __augment_data(df, LENGTH_MAX=1000):
    dataframe = df.copy()
    dataframe_length = len(df)
    while dataframe_length < LENGTH_MAX:
        data = dataframe.sample(1)
        dataframe = dataframe.append(data)
        dataframe_length = len(dataframe)
    return dataframe


def data_augmentation(dataframe):
    for category, length in zip(dataframe['model'], dataframe['len']):
        if length < 1000:
            df = pd.read_csv('../data/dataset/training/' + category)
            new_dataframe = __augment_data(df)
            new_dataframe.to_csv('../data/dataset/training/' + category)
            del new_dataframe
