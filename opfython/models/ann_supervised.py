"""ANN-Supervised Optimum-Path Forest.
"""

import time

import numpy as np

import opfython.math.general as g
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import Heap
from opfython.core import OPF
from opfython.subgraphs.ann import ANNSubgraph
from opfython.algorithms import ann_algorithms

logger = log.get_logger(__name__)


class ANNSupervisedOPF(OPF):
    """A ANNSupervisedOPF which implements the supervised version of OPF classifier with an Approximate-NN subgraph.

    References:
        J. P. Papa and A. X. Falcão. A Learning Algorithm for the Optimum-Path Forest Classifier.
        Graph-Based Representations in Pattern Recognition (2009).

    """

    def __init__(
            self, max_k=1, distance='euclidean', pre_computed_distance=None, ann_params=None):
        """Initialization method.

        Args:
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> KNNSupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(ANNSupervisedOPF, self).__init__(distance, pre_computed_distance)

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        # Training time (training + validation)
        self.fit_time = 0.

        # Test time
        self.pred_time = 0.

        self.ann_params = ann_params

        self.ann_search = None

        # Defining the ann search method
        if ann_params.get('name') == 'annoy':
            self.ann_class = ann_algorithms.Annoy
        elif ann_params.get('name') == 'hnsw':
            self.ann_class = ann_algorithms.HNSW
        else:
            self.ann_class = ann_algorithms.KD_Tree

        logger.info('Class overrided.')

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

        self._max_k = max_k

    def _clustering(self, force_prototype=False):
        """Clusters the subgraph.

        Args:
            force_prototype (bool): Whether clustering should for each class to have at least one prototype.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # For every adjacent node of `i`
            for j in self.subgraph.nodes[i].adjacency:
                # Making sure that variable is an integer
                j = int(j)

                # Checks if node `i` density is equals as node `j` density
                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    # Marks the insertion flag as True
                    insert = True

                    # For every adjacent node of `j`
                    for l in self.subgraph.nodes[j].adjacency:
                        # Making sure that variable is an integer
                        l = int(l)

                        # Checks if it is the same node as `i`
                        if i == l:
                            # If yes, mark insertion flag as False
                            insert = False

                    # If insertion flag is True
                    if insert:
                        # Inserts node `i` in the adjacency list of `j`
                        self.subgraph.nodes[j].adjacency.insert(0, i)

                        dist = self.distance_fn(self.subgraph.nodes[j].features, self.subgraph.nodes[i].features)

                        self.subgraph.nodes[j].adj_distances.insert(0, dist)

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

                # Defines its predicted label as the node's true label
                self.subgraph.nodes[p].predicted_label = self.subgraph.nodes[p].label

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # For every possible adjacent node
            for q in self.subgraph.nodes[p].adjacency:
                # Making sure that variable is an integer
                q = int(q)

                # If its color in the heap is different from `BLACK`
                if h.color[q] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[q].density)

                    # If prototypes should be forced to belong to a class
                    if force_prototype:
                        # Checks if nodes `p` and `q` labels are different
                        if self.subgraph.nodes[p].label != self.subgraph.nodes[q].label:
                            # If yes, define current cost as minimum value possible
                            current_cost = -c.FLOAT_MAX

                    # If current cost is bigger than heap's cost
                    if current_cost > h.cost[q]:
                        # Apply `q` predecessor as `p`
                        self.subgraph.nodes[q].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[p].predicted_label

                        # Updates node `q` on the heap with the current cost
                        h.update(q, current_cost)

    def _learn(self, X_train, Y_train, I_train, X_val, Y_val, I_val):
        """Learns the best `k` value over the validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            I_val (np.array): Array of validation indexes.

        """

        logger.info('Learning best `k` value ...')

        # Creating a subgraph
        self.subgraph = ANNSubgraph(X_train, Y_train, I_train)

        # Defining initial maximum accuracy as 0
        max_acc = 0.

        best_k = 1

        # For every possible `k` value
        for k in range(1, self.max_k + 1):
            # Gathers current `k` as subgraph's best `k`
            self.subgraph.best_k = k

            # Initiating the ANN method to perform Approximate Nearest Neighbors search
            if self.ann_params.get('name') == 'hnsw':
                self.ann_params['ef'] = k

            self.ann_search = self.ann_class(self.ann_params)

            # Build the ANN index
            self.ann_search.fit(X_train)

            # Calculate the arcs using the current `k` value
            self.subgraph.build_arcs(k, self.ann_search)

            # Calculate the p.d.f. using the current `k` value
            # self.subgraph.calculate_pdf(
            #     k, self.distance_fn, self.pre_computed_distance, self.pre_distances)
            self.subgraph.calc_pdf(k, self.distance_fn)

            # Clusters the subgraph
            self._clustering()

            # Calculate the predictions over the validation set
            preds = self.predict(X_val, I_val)

            # Calculating the accuracy
            acc = g.opf_accuracy(Y_val, preds)

            # If accuracy is better than maximum accuracy
            if acc > max_acc:
                # Replaces the maximum accuracy value
                max_acc = acc

                # Defines current `k` as the best `k` value
                best_k = k

            logger.info('Accuracy over k = %d: %s', k, acc)

            # Destroy the arcs
            self.subgraph.destroy_arcs()

        # Applying the best k to the subgraph's property
        self.subgraph.best_k = best_k

    def fit(self, X_train, Y_train, X_val, Y_val, I_train=None, I_val=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            I_train (np.array): Array of training indexes.
            I_val (np.array): Array of validation indexes.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer
        start = time.time()

        # Performing the learning process in order to find the best `k` value
        self._learn(X_train, Y_train, I_train, X_val, Y_val, I_val)

        # if self.ann_params.get('name') == 'hnsw':
        #     self.ann_params['ef'] = self.subgraph.best_k

        # Creating Index to perform Approximate Nearest Neighbors search
        self.ann_search = self.ann_class(self.ann_params)

        self.ann_search.fit(X_train)

        # Creating arcs with the best `k` value
        self.subgraph.build_arcs(self.subgraph.best_k, self.ann_search)

        # Calculating p.d.f. with the best `k` value
        self.subgraph.calc_pdf(self.subgraph.best_k, self.distance_fn)
        # self.subgraph.calculate_pdf(
        #     self.subgraph.best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Clustering subgraph forcing each class to have at least one prototype
        self._clustering(force_prototype=True)

        # Destroying arcs
        self.subgraph.destroy_arcs()

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        self.fit_time = end - start

        logger.info('Classifier has been fitted with k = %d.', self.subgraph.best_k)
        logger.info(f'Training time: {self.fit_time:.4f} seconds.')

    def predict(self, X_test, I_test=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_test (np.array): Array of features.
            I_test (np.array): Array of indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = ANNSubgraph(X_test, I=I_test)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # self.ann_search.load_index('ann_index.bin', max_samples=self.ann_search.n_samples)

        # Creating an array of distances
        # distances = np.zeros(best_k + 1)

        # Creating an array of nearest neighbours indexes
        # neighbours_idx = np.zeros(best_k + 1)

        # For every possible prediction node
        for i in range(pred_subgraph.n_nodes):
            # For every possible trained node
            neighbors_idx, distances = self.ann_search.query(pred_subgraph.nodes[i].features, best_k)

            density = np.sum(np.exp(-np.array(distances) / self.subgraph.constant))

            # Gather its mean value
            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

            neighbor_costs = [self.subgraph.nodes[neighbor].cost for neighbor in neighbors_idx]

            # Calculate the temporary cost
            temp_cost = np.minimum(neighbor_costs, [density])

            # Select the maximum cost among node's neighbors
            k = np.argmax(temp_cost)

            # Gathers the node's neighbor
            neighbor = int(neighbors_idx[k])

            # Propagates the predicted label from the neighbour
            pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbor].predicted_label

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        self.pred_time = end - start

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {self.pred_time:.4f} seconds.')

        return preds
