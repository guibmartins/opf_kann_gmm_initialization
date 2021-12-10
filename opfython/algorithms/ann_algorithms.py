import numpy as np
import annoy
import hnswlib
# import scann

from sklearn.neighbors import KDTree, NearestNeighbors
from opfython.utils import exception as e


# Base class for implementation of fast ANN search algorithms
class ApproximateNNSearch:

    def __init__(self, params=None):

        if params is None:
            params = {}

        self.params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params: dict):

        if not isinstance(params, dict):
            raise e.TypeError('`params` should be a dictionary')

        self._params = params

    def set_default_params(self):
        raise NotImplementedError

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def query(self, sample: np.ndarray, k: int):
        raise NotImplementedError


class NNeighbors(ApproximateNNSearch):

    def __init__(self, params: dict):

        super(NNeighbors, self).__init__(params)

        if len(params) == 0:

            self.set_default_params()

        else:

            if params.get('n_neighbors') is None:
                self.params['n_neighbors'] = 1

            if params.get('algorithm') is None:
                self.params['algorithm'] = 'brute'

            if params.get('distance') is None:
                self.params['distance'] = 'minkowski'

            if params.get('p') is None:
                self.params['p'] = 2

            if params.get('leaf_size') is None:
                self.params['leaf_size'] = 30

            if params.get('n_jobs') is None:
                self.params['n_jobs'] = -1

        self.nneighbors = NearestNeighbors(
            n_neighbors=self.params.get('n_neighbors'),
            algorithm=self.params.get('algorithm'),
            leaf_size=self.params.get('leaf_size'),
            metric=self.params.get('distance'),
            p=self.params.get('p'),
            n_jobs=self.params.get('n_jobs')
        )

    @property
    def nneighbors(self):
        return self._nneighbors

    @nneighbors.setter
    def nneighbors(self, nneighbors: NearestNeighbors):
        self._nneighbors = nneighbors

    def set_default_params(self):

        self.params.update({
            'n_neighbors': 1,
            'leaf_size': 30,
            'algorithm': 'brute',
            'distance': 'minkowski',
            'p': 2,
            'n_jobs': -1
        })

    def fit(self, X: np.ndarray):

        self.nneighbors.fit(X)

    def query(self, data: np.ndarray, k: int = 1):

        distances, indices = self.nneighbors.kneighbors(
            data[np.newaxis], n_neighbors=k, return_distance=True)

        return indices.squeeze().tolist(), distances.squeeze().tolist()


class Annoy(ApproximateNNSearch):

    def __init__(self, params: dict):

        # Override its parent class with the receiving arguments
        super(Annoy, self).__init__(params)

        if len(params) == 0:

            self.set_default_params()

        else:

            if params.get('distance') is None:
                self.params['distance'] = 'euclidean'

            if params.get('n_trees') is None:
                self.params['n_trees'] = 2

            if params.get('n_jobs') is None:
                self.params['n_jobs'] = -1

            self.tree = annoy.AnnoyIndex(
                params.get('n_features'), params.get('distance')
            )

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree: annoy.AnnoyIndex):
        self._tree = tree

    def set_default_params(self):

        self.params.update({
            'distance': 'euclidean',
            'n_trees': 2,
            'n_jobs': -1
        })

    def fit(self, X: np.ndarray, save_file: bool = False):

        for i in range(X.shape[0]):
            self.tree.add_item(i, X[i])

        self.tree.build(self.params.get('n_trees'), n_jobs=self.params.get('n_jobs'))

        if save_file:
            self.tree.save('annoy_index.bin')

    def query(self, data: np.ndarray, k: int = 1):

        indices, distances = self.tree.get_nns_by_vector(data, k, include_distances=True)
        return indices, distances


class HNSW(ApproximateNNSearch):

    def __init__(self, params: dict):

        # Override its parent class with the receiving arguments
        super(HNSW, self).__init__(params)

        if not len(params):
            self.set_default_params()

        else:
            if params.get('space') is None:
                self.params['distance'] = 'l2'

            if params.get('ef') is None:
                self.params['ef'] = 10

            if params.get('M') is None:
                self.params['M'] = 16

            if params.get('random_state') is None:
                self.params['random_state'] = 100

            if params.get('n_jobs') is None:
                self.params['n_jobs'] = -1

        # possible space options are l2, cosine or ip (inner product)
        self.graph = hnswlib.Index(space=params.get('distance'), dim=int(params.get('n_features')))

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph: hnswlib.Index):
        self._graph = graph

    def set_default_params(self):

        self.params.update({
            'name': 'hnsw',
            'distance': 'l2',
            'ef': 100,
            'ef_construction': 200,
            'M': 16,
            'random_state': None,
            'n_jobs': -1
        })

    def fit(self, X: np.ndarray, save_file: bool = False):

        # Initiate index - the maximum number of elements should be known
        self.graph.init_index(
            max_elements=self.params.get('n_samples'),
            ef_construction=self.params.get('ef_construction'),
            M=self.params.get('M'),
            random_seed=self.params.get('random_state')
        )

        n_labels = np.arange(self.params.get('n_samples'))

        # Element insertion
        self.graph.add_items(X, n_labels, num_threads=self.params.get('n_jobs'))

        # Controlling the search recall by setting ef
        # ef should always be > k and < n_samples
        self.set_ef(self.params.get('ef'))

        if save_file:
            self.save_index('hnsw_index.bin')

    def query(self, data: np.ndarray, k: int = 1, n_jobs=-1):

        indices, distances = self.graph.knn_query(data, k, num_threads=n_jobs)
        return list(indices[0]), list(distances[0])

    def save_index(self, path_file: str):

        self.graph.save_index(path_file)
        self.graph = None

    def load_index(self, path_file: str, max_samples: int):

        self.graph = hnswlib.Index(space=self.params.get('distance'), dim=self.params.get('n_features'))
        self.graph.load_index(path_file, max_elements=max_samples)
        self.graph.set_ef(max_samples)

    def set_ef(self, ef: int):

        self.graph.set_ef(ef)


class KD_Tree(ApproximateNNSearch):

    def __init__(self, params: dict):

        # Override its parent class with the receiving arguments
        super(KD_Tree, self).__init__(params)

        if len(params) == 0:

            self.set_default_params()

        else:

            if params.get('leaf_size') is None:
                self.params['leaf_size'] = 40

            if params.get('distance') is None:
                self.params['distance'] = 'minkowski'

        self.tree = None

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree: KDTree):
        self._tree = tree

    def set_default_params(self):

        self.params.update({'leaf_size': 40, 'distance': 'minkowski'})

    def fit(self, X: np.ndarray, save_file: bool = False):

        self.tree = KDTree(
            X,
            leaf_size=self.params.get('leaf_size'),
            metric=self.params.get('distance')
        )

    def query(self, sample: np.ndarray, k: int, dual_tree: bool = False):

        distances, indices = self.tree.query(sample[np.newaxis], k=k, dualtree=dual_tree)
        return indices.squeeze().tolist(), distances.squeeze().tolist()


class Scann(ApproximateNNSearch):

    def __init__(self, params):
        super().__init__(params)

    def set_default_params(self):
        pass

    def fit(self, X):
        pass

    def query(self, sample, k):
        pass
