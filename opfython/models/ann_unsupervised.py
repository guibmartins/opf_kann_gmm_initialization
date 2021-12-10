"""Unsupervised Optimum-Path Forest.
"""

import time
import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import OPF
from opfython.core import Heap
from opfython.subgraphs.ann import ANNSubgraph
from opfython.algorithms import ann_algorithms

logger = log.get_logger(__name__)


class ANNUnsupervisedOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
            self, min_k=1, max_k=1, distance='squared_euclidean', eliminate_maxima=None,
            ann_params=None, pre_computed_distance=None, pdf_function='gaussian', weight_pdf=False):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        # logger.info('Overriding class: OPF -> ANNUnsupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(ANNUnsupervisedOPF, self).__init__(distance, pre_computed_distance)

        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        # Defining the minimum `k` value for cutting the subgraph
        self.eliminate_maxima = {} if eliminate_maxima is None else eliminate_maxima

        self.min_cut = c.FLOAT_MAX

        self.ann_params = ann_params

        self.ann_search = None

        self.fit_time = 0.

        self.pred_time = 0.

        self.pdf_function = pdf_function

        self.weight_pdf = weight_pdf

        # Defining the ann search method
        if ann_params.get('name') == 'annoy':
            self.ann_class = ann_algorithms.Annoy
        elif ann_params.get('name') == 'hnsw':
            self.ann_class = ann_algorithms.HNSW
        elif ann_params.get('name') == 'kdtree':
            self.ann_class = ann_algorithms.KD_Tree
        else:
            self.ann_class = ann_algorithms.NNeighbors

        logger.info(f"NN Search algorithm: {ann_params.get('name')}\tPDF function: {pdf_function}")
        # logger.info('Class overrided.')

    @property
    def min_k(self):
        """int: Minimum `k` value for cutting the subgraph.

        """

        return self._min_k

    @min_k.setter
    def min_k(self, min_k):
        if not isinstance(min_k, int):
            raise e.TypeError('`min_k` should be an integer')
        if min_k < 1:
            raise e.ValueError('`min_k` should be >= 1')

        self._min_k = min_k

    @property
    def max_k(self):
        """int: Maximum `k` value for cutting the subgraph.

        """

        return self._max_k

    @max_k.setter
    def max_k(self, max_k):
        if not isinstance(max_k, int):
            raise e.TypeError('`max_k` should be an integer')
        if max_k < 1:
            raise e.ValueError('`max_k` should be >= 1')
        if max_k < self.min_k:
            raise e.ValueError('`max_k` should be >= `min_k`')

        self._max_k = max_k

    def _clustering(self, n_neighbours):
        """Clusters the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):

            # For every possible `k` value
            for k in range(n_neighbours):
                # Gathers node `i` adjacent node
                j = int(self.subgraph.nodes[i].adjacency[k])

                # If both nodes' density are equal
                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    # Turns on the insertion flag
                    insert = True

                    # For every possible `l` value
                    for l in range(n_neighbours):

                        # Gathers node `j` adjacent node
                        adj = int(self.subgraph.nodes[j].adjacency[l])

                        # If the nodes are the same
                        if i == adj:
                            # Turns off the insertion flag
                            insert = False

                        # If it is supposed to be inserted
                        if insert:

                            dist = self.distance_fn(
                                self.subgraph.nodes[j].features, self.subgraph.nodes[i].features)

                            # Inserts node `i` in the adjacency list of `j`
                            self.subgraph.nodes[j].adjacency.insert(0, i)

                            # Inserts distance of node `i` in the distance adjacency list of `j`
                            self.subgraph.nodes[j].adj_distances.insert(0, dist)

                            # Increments the amount of adjacent nodes
                            self.subgraph.nodes[j].n_plateaus += 1

        # Creating a maximum heap
        h = Heap(size=self.subgraph.n_nodes, policy='max')

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # Updates the node's cost on the heap
            h.cost[i] = self.subgraph.nodes[i].cost

            # Defines node's `i` predecessor as NIL
            self.subgraph.nodes[i].pred = c.NIL

            # And its root as its same identifier
            self.subgraph.nodes[i].root = i

            # Inserts the node in the heap
            h.insert(i)

        # Defining an `l` counter
        l = 0

        # While the heap is not empty
        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # If the node's predecessor is NIL
            if self.subgraph.nodes[p].pred == c.NIL:
                # Updates its cost on the heap
                h.cost[p] = self.subgraph.nodes[p].density

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].cluster_label = l

                # Increments the cluster identifier
                l += 1

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_plateaus + n_neighbours

            # For every possible adjacent node
            for k in range(n_adjacents):
                # Gathers the adjacent identifier
                q = int(self.subgraph.nodes[p].adjacency[k])

                # If its color in the heap is different from `BLACK`
                if h.color[q] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[q].density)

                    # If temporary cost is bigger than heap's cost
                    if current_cost > h.cost[q]:
                        # Apply `q` predecessor as `p`
                        self.subgraph.nodes[q].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[q].cluster_label = self.subgraph.nodes[p].cluster_label

                        # Updates the heap `q` node and the current cost
                        h.update(q, current_cost)

        # The final number of clusters will be equal to `l`
        self.subgraph.n_clusters = l

    def _normalized_cut(self, n_neighbours):
        """Performs a normalized cut over the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours (int): Number of neighbours to be used.

        Returns:
            The value of the normalized cut.

        """

        # Defining an array to represent the internal cluster distances
        internal_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining an array to represent the external cluster distances
        external_cluster = np.zeros(self.subgraph.n_clusters)

        # Defining the cut value
        cut = 0.0

        # For every possible node
        for i in range(self.subgraph.n_nodes):

            # Calculates its number of adjacent nodes
            n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            distance = self.subgraph.nodes[i].adj_distances

            # For every possible adjacent node
            for k in range(n_adjacents):

                # Gathers its adjacent node identifier
                j = int(self.subgraph.nodes[i].adjacency[k])

                # If distance is bigger than 0
                if distance[k] > 0:
                    # If nodes belongs to the same clusters
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        # Increments the internal cluster distance
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

                    # If nodes belongs to distinct clusters
                    else:
                        # Increments the external cluster distance
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1. / float(distance[k])

        # For every possible cluster
        for l in range(self.subgraph.n_clusters):
            # If the sum of internal and external clusters is bigger than 0
            if internal_cluster[l] + external_cluster[l] > 0.0:
                # Increments the value of the cut
                cut += external_cluster[l] / (internal_cluster[l] + external_cluster[l])

        return cut

    def _best_minimum_cut(self, min_k, max_k):
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            min_k (int): Minimum value of k.
            max_k (int): Maximum value of k.

        """

        # logger.debug(
        #     'Calculating the best minimum cut within [%d, %d] ...', min_k, max_k)

        # Calculates the maximum possible distances
        max_distances = self.subgraph.build_arcs(max_k, self.ann_search)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX

        best_k = max_k

        # For every possible value of `k`
        for k in range(min_k, max_k + 1):

            if min_cut == 0.0:
                break

            # If minimum cut is different than zero
            # if min_cut != 0.0:

            # Gathers the subgraph's density
            self.subgraph.density = max_distances[k - 1]

            # Gathers current `k` as the subgraph's best `k` value
            self.subgraph.best_k = k

            # Calculates the p.d.f.
            self.subgraph.calc_pdf(k, self.distance_fn, self.ann_search)

            # Clustering with current `k` value
            self._clustering(k)

            # Performs the normalized cut with current `k` value
            cut = self._normalized_cut(k)

            # If the cut's cost is smaller than minimum cut
            if cut < min_cut:
                # Replace its value
                min_cut = cut

                # And defines a new best `k` value
                best_k = k

        self.min_cut = min_cut

        # Destroy current arcs
        self.subgraph.destroy_arcs()

        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.build_arcs(best_k, self.ann_search)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calc_pdf(best_k, self.distance_fn, self.ann_search)

        logger.debug(f'Best: {best_k} | Minimum cut: {min_cut: .4f}.')

    def fit(self, X_train, Y_train=None, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Clustering with classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = ANNSubgraph(X_train, Y_train, I_train, pdf_fn=self.pdf_function, weight_pdf=self.weight_pdf)

        self.ann_params.update({'n_samples': X_train.shape[0], 'n_features': X_train.shape[1]})

        # Initiating the ANN method to perform Approximate Nearest Neighbors search
        if self.ann_params.get('name') == 'hnsw':
            self.ann_params['ef'] = self.max_k + 1
            self.ann_params['ef_construction'] = self.max_k * 100

        if self.ann_params.get('name') == 'knn':
            self.ann_params['n_neighbors'] = self.max_k

        self.ann_search = self.ann_class(self.ann_params)

        # Build the ANN index
        self.ann_search.fit(X_train)

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        if len(self.eliminate_maxima) > 0:

            if list(self.eliminate_maxima.keys())[0] in ['height', 'area', 'volume']:

                key, value = self.eliminate_maxima.popitem()

                if key == 'height':
                    self.subgraph.eliminate_maxima_height(value)
                elif key == 'area':
                    self.subgraph.eliminate_maxima_area(value)
                else:
                    self.subgraph.eliminate_maxima_volume(value)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        self.fit_time = end - start

        # logger.info('Classifier has been clustered with.')
        logger.info(f'Number of clusters: {self.subgraph.n_clusters}.')
        # logger.info(f'Clustering time: {self.fit_time : .4f} seconds.')

    def predict(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation features.
            I_val (np.array): Array of validation indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a knn-subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('ANNSubgraph has not been properly created')

        # Checks if knn-subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly clustered')

        # logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = ANNSubgraph(X_val, I=I_val, pdf_fn=self.pdf_function, weight_pdf=self.weight_pdf)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # Creating an array of distances
        # distances = np.zeros(best_k + 1)

        # Creating an array of nearest neighbours indexes
        # neighbours_idx = np.zeros(best_k + 1)

        # For every possible prediction node
        for i in range(pred_subgraph.n_nodes):

            # For every possible trained node
            neighbors_idx, distances = self.ann_search.query(pred_subgraph.nodes[i].features, best_k)

            density = np.sum(np.exp(-np.array(distances) / float(self.subgraph.constant)))

            # Gather its mean value
            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1.

            neighbor_costs = [self.subgraph.nodes[neighbor].cost for neighbor in neighbors_idx]

            # Calculate the temporary cost
            temp_cost = np.minimum(neighbor_costs, [density])

            # Select the maximum cost among node's neighbors
            k = np.argmax(temp_cost)

            # Gathers the node's neighbor
            neighbor = int(neighbors_idx[k])

            # Propagates the predicted label from the neighbour
            pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbor].predicted_label

            # Propagates the cluster label from the neighbour
            pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[neighbor].cluster_label

            del neighbor_costs
            del neighbor

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Creating the list of clusters
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        self.pred_time = end - start

        # logger.info('Data has been predicted.')
        # logger.info(f'Prediction time: {self.pred_time : .4f} seconds.')

        return preds, clusters

    def fit_predict(self, X_train, Y_train=None, I_train=None):

        self.fit(X_train, Y_train=Y_train, I_train=I_train)

        return np.array([node.cluster_label for node in self.subgraph.nodes])

    def propagate_labels(self):
        """Runs through the clusters and propagate the clusters roots labels to the samples.

        """

        # logger.info('Assigning predicted labels from clusters ...')

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # Gathers the root from the node
            root = self.subgraph.nodes[i].root

            # If the root is the same as node's identifier
            if root == i:
                # Apply the predicted label as node's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

            # If the root is different from node's identifier
            else:
                # Apply the predicted label as the root's label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        # logger.info('Labels assigned.')


class HierarchicalUnsupervisedOPF(ANNUnsupervisedOPF):
    """An Hierarchical OPF clustering which implements the unsupervised version of OPF classifier.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """
    def __init__(self, min_k=1, max_k=1, distance='squared_euclidean', elim_maxima=None,
                 ann_params=None, pre_computed_distance=None, pdf_fn='gaussian', weight_pdf=False):
        """Initialization method.

        Args:
            min_k (int): Minimum `k` value for cutting the subgraph.
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """
        # Override its parent class with the receiving arguments
        super(HierarchicalUnsupervisedOPF, self).__init__(min_k, max_k, distance, elim_maxima,
                                                          ann_params, pre_computed_distance, pdf_fn, weight_pdf)

        self.sg_hierarchy = []

        self.clusters_hierarchy = []

        self.fit_time = 0.

        self.pred_time = 0.

    @property
    def sg_hierarchy(self):
        return self._sg_hierarchy

    @sg_hierarchy.setter
    def sg_hierarchy(self, sg_hierarchy):
        if not isinstance(sg_hierarchy, list):
            raise e.TypeError('`sg_hierarchy` should be a list')

        self._sg_hierarchy = sg_hierarchy

    def fit(self, X_train, Y_train=None, I_train=None):

        print('Clustering with Hierarchical OPF...')

        layer = 0
        max_k = self.max_k
        X_layer = X_train.copy()
        n_clusters = X_train.shape[0]
        self.ann_params.update(
            {'n_samples': X_train.shape[0],
             'n_features': X_train.shape[1]})

        tic = time.time()

        while n_clusters > 1:

            print('Clustering on layer', layer, '...')

            self.ann_params.update({'n_samples': X_layer.shape[0]})

            opf = ANNUnsupervisedOPF(max_k=max_k, ann_params=self.ann_params,
                                     pdf_function=self.pdf_function, weight_pdf=self.weight_pdf)

            clusters = opf.fit_predict(X_layer)

            n_clusters = opf.subgraph.n_clusters

            # Stack the prototypes to compose the training set concerning layer l
            X_layer = [i.features
                       for label in range(n_clusters)
                       for i in opf.subgraph.nodes
                       if (i.pred == -1 and i.cluster_label == label)]

            X_layer = np.asfarray(X_layer)

            self.sg_hierarchy.append(opf)
            self.clusters_hierarchy.append(clusters)

            # Redefine the max_k considering the current clustering results
            max_k = int(np.min([opf.subgraph.best_k, X_layer.shape[0] - 1]))

            layer += 1

        self.fit_time = time.time() - tic
        print(f'Clustering time: {self.fit_time:.4f} sec.')

    def propagate_clusters(self, clusters_hierarchy: list):

        if len(clusters_hierarchy) == 1:
            return clusters_hierarchy.pop()

        # Get index of the last layer
        L = len(clusters_hierarchy) - 1

        print('Propagate labels from layer', L, 'to', L - 1, '...')

        preds = clusters_hierarchy[L - 1].copy()

        for i_root, label in enumerate(clusters_hierarchy[L]):
            idx = np.flatnonzero(clusters_hierarchy[L - 1] == i_root)
            preds[idx] = label

        clusters_hierarchy[L - 1] = preds
        clusters_hierarchy.pop()

        return self.propagate_clusters(clusters_hierarchy)
