from keras.backend import clear_session

from source.model.canonizador import Canonizador
from source.data_processing.utils.utils import format_path_name
from source.data_processing.utils.utils import configure_paths
from source.data_processing.utils.utils import load_tree
from source.data_processing.utils.utils import have_children


class DeepTreeModel:
    def __init__(self, hierarchy_level):
        self.hierarchy_level = hierarchy_level

    def __traverse_inferential_tree(self, raw_data, first_inference, hierarchy_level=0):
        tree = load_tree(format_path_name(root_dir='../../data/trees/', model_name=first_inference,
                                          mode='.pickle'))

        while hierarchy_level < self.hierarchy_level:
            if have_children(tree):
                model_weight_path, class_path, path_model_architecture = configure_paths(first_inference,
                                                                                         hierarchy_level - 1)
                model_hierarchy = Canonizador(path_model_weights=model_weight_path,
                                              path_class=class_path,
                                              path_model_architecture=path_model_architecture)

                inference = model_hierarchy.predict_model(raw_data)
                first_inference += '_' + inference
                hierarchy_level += 1
                tree = tree.subtree(first_inference)

                clear_session()
                del model_hierarchy
            else:
                inference = [child._tag for child in tree.children(tree.root)]
                if len(inference) > 0:
                    first_inference += '_' + inference[0]
                break
        return first_inference.replace('_', '/')

    def make_inference(self, raw_data):
        canonizador_init = Canonizador(path_model_weights='../../data/trained_models/weights/model_0.h5',
                                       path_class='../../data/labels/model_0.txt',
                                       path_model_architecture='../../data/trained_models/architectures/'
                                                               'model_0.json')

        first_inference = canonizador_init.predict_model(raw_data)
        hierarchy_level = 1

        clear_session()
        del canonizador_init

        if self.hierarchy_level == hierarchy_level:
            return first_inference
        final_inference = self.__traverse_inferential_tree(raw_data, first_inference,
                                                           hierarchy_level)
        return final_inference
