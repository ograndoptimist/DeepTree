import pandas as pd
from keras.backend import clear_session

from source.model.deep_tree_model import DeepTreeModel


def filter_by_hierarchy(hierarchy_level, df):
    df['length'] = df.struct.apply(lambda x: len(x.split('/')))
    df = df[df['length'] >= hierarchy_level]
    df.struct = df.struct.apply(lambda x: '/'.join(x.split('/')[:hierarchy_level]))
    return df[['termo', 'struct']]


def build_dataframe(df):
    df1, df2, df3, df4 = filter_by_hierarchy(hierarchy_level=1, df=df), \
                         filter_by_hierarchy(hierarchy_level=2, df=df), \
                         filter_by_hierarchy(hierarchy_level=3, df=df), \
                         filter_by_hierarchy(hierarchy_level=4, df=df)
    return {1: df1, 2: df2, 3: df3,
            4: df4}


def build_metrics(dataframe_path, set_level):
    df = pd.read_csv(dataframe_path)
    dfs = build_dataframe(df)
    del df

    for level, df in dfs.items():
        if level == set_level:
            estimator = DeepTreeModel(hierarchy_level=level)
            measure = pd.DataFrame(columns=['output', 'target'])
            measure['target'] = df['struct']
            measure['output'] = df.termo.apply(lambda x: print(estimator.make_inference(x)))

            # measure.to_csv('../../data/metrics/output_model_level_' + str(level) + '.csv')

            clear_session()
            del estimator

        if set_level == level:
            break


if __name__ == '__main__':
    build_metrics(dataframe_path='../../data/metrics/test_data.csv', set_level=1)
