from treelib import Tree
from numpy import nan


def tree_node(tree, list_split, length_list, parent_category, number_iter=0):
    """
        Insert nodes on the tree.
    """
    if number_iter == length_list:
        return None
    else:
        identifier = parent_category + '_' + list_split[number_iter]
        if identifier not in tree:
            tree.create_node(parent=parent_category, identifier=identifier, tag=list_split[number_iter], data=0)
            return tree_node(tree=tree, list_split=list_split, length_list=length_list, number_iter=number_iter + 1,
                             parent_category=identifier)
        else:
            return tree_node(tree=tree, list_split=list_split, length_list=length_list, number_iter=number_iter + 1,
                             parent_category=identifier)


def build_tree(dataset, category, column_output):
    """
        Build the final tree corresponding to the category.
    """
    dataset = build_dataset(dataset, category, column_output)

    tree = Tree()
    tree.create_node(identifier=category, data=0)

    for item in dataset:
        if item != '':
            lista_split = item.split('/')
            tree_node(tree=tree, list_split=lista_split, length_list=len(lista_split),
                      number_iter=0, parent_category=category)
    return tree


def build_dataset(dataset, category, column_output):
    """
        Split the dataset to only contain the category considered.
    """
    dataset = filter(lambda x: x.split('/')[0] == category, dataset[column_output].unique())
    dataset = map(lambda x: '/'.join(x.split('/')[1:]), dataset)
    return list(dataset)


def get_crm(tree_list, category):
    for index in range(len(tree_list)):
        if list(tree_list[index].nodes.keys())[0] == category:
            return tree_list[index]


def parse_tree(tree, root, level, parent_max, acumuled=[], tree_fix=None):
    """
        Traverses the tree using the breadth first strategy to determine
        which categories need models and its children.
    """
    if tree_fix is None:
        tree_fix = tree
    if tree.root is not None:
        if len(tree.children(root)) > 1:
            children_tag = [child.tag for child in tree.children(root)]
            children_identifier = [child._identifier for child in tree.children(root)]
            teste = {'parent_category': parent_max, 'root': tree.root, 'level': tree_fix.depth(tree.root),
                     'children': sorted(children_tag)}
            acumuled.append(teste)
            [parse_tree(tree.subtree(child), tree.subtree(child).root, level, parent_max, acumuled, tree_fix)
             for child in children_identifier]
    return acumuled


def pick_category_dataframe(dataframe, category, column_output):
    """
        Returns a dataframe that only contains the category considered.
        ::params:
            ::dataframe:
            ::category: the category on hierarchy level 0

        Example:
                A dataframe that contains:
                >>> dataframe
                query_string                                output
                toalha de banho                             cama-mesa-e-banho/jogo-de-banho
                microondas de embutir 220v electrolux       eletrodomesticos/micro-ondas/micro-ondas-de-em...
                teanina                                     suplementos-e-vitaminas/energia-e-resistencia
                rack branco                                 moveis/rack-estante-e-painel/rack
                jardim encantado                            artigos-de-festas/decoracao-de-festa/painel-de...
                pneu aro 16                                 automotivo/pneus/pneu-de-passeio
                lipo                                        suplementos-e-vitaminas/emagrecimento
                tela iphone 5s original                     celulares-e-smartphones/pecas-para-celular/tel...
                kit de cama 7 pecas                         cama-mesa-e-banho/colchas/colcha-casal

                After calling the function, should return:
                >>> pick_category_dataframe(dataframe, 'cama-mesa-e-banho')
                query_string                                output
                toalha de banho                             cama-mesa-e-banho/jogo-de-banho
                kit de cama 7 pecas                         cama-mesa-e-banho/colchas/colcha-casal
    """
    new_dataframe = dataframe.copy()
    new_dataframe[column_output] = new_dataframe[column_output].apply(lambda x: x if x.split('/')[0] == category else nan)
    return new_dataframe.dropna()


def filter_dataframe_by_len(dataframe, level, column_output):
    """
        Returns a new dataframe filtered by its length.
        ::params:
            ::level:

        Example:
                A dataframe that contains:
                >>> dataframe
                query_string                                output
                toalha de banho                             cama-mesa-e-banho/jogo-de-banho
                kit de cama 7 pecas                         cama-mesa-e-banho/colchas/colcha-casal

                After calling the function, should return:
                >>> filter_dataframe_by_len(dataframe, 1)
                query_string                                output
                kit de cama 7 pecas                         cama-mesa-e-banho/colchas/colcha-casal
    """
    new_dataframe = dataframe.copy()
    new_dataframe['len'] = new_dataframe[column_output].apply(lambda x: len(x.split('/')))
    new_dataframe = new_dataframe[new_dataframe['len'] > level + 1]
    new_dataframe = new_dataframe.drop(['len'], axis=1)
    return new_dataframe


def dataset_next_hierarchy(dataframe, category, level, column_output):
    """
        Returns a new dataframe containing the next level of hierarchy.
        ::params:
            ::dataframe:
            ::category: a string containing the
            ::category_level: an int that reflects the level of hierarchy.

        Example:
                A dataframe that contains:
                >>> dataframe
                query_string                                output
                toalha de banho                             cama-mesa-e-banho/jogo-de-banho
                kit de cama 7 pecas                         cama-mesa-e-banho/colchas/colcha-casal

                After calling the function, should return:
                >>> pick_category_dataframe(dataframe, 'cama-mesa-e-banho', 0)
                query_string                                output
                toalha de banho                             jogo-de-banho
                kit de cama 7 pecas                         colchas
    """
    new_dataframe = dataframe.copy()
    new_dataframe = filter_dataframe_by_len(new_dataframe, level, column_output)
    new_dataframe[column_output] = new_dataframe[column_output].apply(lambda x: x.split('/')[level + 1]
    if x.split('/')[level] == category else nan)
    new_dataframe = new_dataframe.dropna()
    return new_dataframe


def split_data(dataframe, level, parent_category, category, column_output):
    """
        Split the dataset to only contain the category considered with
        respect to its hierarchy on the dataframe.
    """
    new_dataframe = dataframe.copy()
    new_dataframe = pick_category_dataframe(new_dataframe, parent_category, column_output)
    new_dataframe = dataset_next_hierarchy(new_dataframe, category, level, column_output)
    return new_dataframe


def search_models_tree(tree_list, category_list):
    """
        Search through the tree the nodes that need models and returns a dict containing
        the information to later train the model.
    """
    final = []
    for category in category_list:
        tree = get_crm(tree_list, category)
        final.append(parse_tree(tree=tree, root=category, level=tree.depth(), parent_max=category, acumuled=[]))
    return final


def fix_dataset(dataframe, children, column_output):
    df = dataframe.copy()
    df.index = range(len(df))
    for index in range(len(df)):
        if df.loc[index, column_output] not in children:
            df.loc[index, column_output] = nan
    df = df.dropna()
    df.index = range(len(df))
    return df


def build_dataframe(tree_list, category_list, dataframe, column_output, queue_models):
    """
        Saves the datasets corresponding to each node of the tree that has children.
        The saved dataset will be used to train its respectives models.
    """
    dataset_paths = []
    for category_root in queue_models:
        print('\tInitializing {0} dataset...'.format(str(category_root['root']) + '_level_'
                                                     + str(category_root['level'])))
        data_acc = split_data(dataframe=dataframe, level=category_root['level'],
                              parent_category=category_root['parent_category'],
                              category=category_root['root'].split('_')[-1], column_output=column_output)
        data_acc = fix_dataset(dataframe=data_acc, children=category_root['children'], column_output=column_output)
        if len(data_acc) == 0:
            print("ERROR FOR ", category_root['root'].split('_')[-1])
            print("PARENT: ", category_root['parent_category'])
            print("LEVEL: ", category_root['level'])
            print()
        else:
            save_local = '../data/dataset/training/' + str(category_root['root']) + '_level_' \
                         + str(category_root['level']) + '.csv'
            dataset_paths.append({'model_name': str(category_root['root']) + '_level_' + str(category_root['level']),
                                  'len': len(data_acc)})
            data_acc.to_csv(save_local)
        print('\tFinishing {0} dataset...'.format(str(category_root['root']) + '_level_'
                                                  + str(category_root['level'])))
        del data_acc
    return dataset_paths
