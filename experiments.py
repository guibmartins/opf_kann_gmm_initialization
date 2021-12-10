import sys
import time
import datetime as dt
import numpy as np
import pandas as pd

import dataset as d
import parameters as p
import gaussian_estimation as ge

# Extrinsic clustering measures
# from c_index import calc_c_index
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from sklearn.metrics.cluster import adjusted_rand_score

# Intrinsic clustering measures
from sklearn.metrics.cluster import davies_bouldin_score

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from opfython.models.ann_unsupervised import ANNUnsupervisedOPF
from opfython.utils import constants


def get_datetime():
    current_date = dt.datetime.now()
    return current_date.strftime("%b-%d-%Y_%H-%M-%S-%f").lower()


def get_max_k(X_val: np.ndarray, max_k: int, ann_name='knn', custom_state=None):

    ann = p.prm.get(ann_name).copy()

    if ann.get('name') == 'knn':
        ann.update({'n_neighbors': max_k})

    if ann.get('name') in ['kdtree', 'hnsw']:
        ann.update({'random_state': custom_state})

    best_max_k = 5
    min_score = constants.FLOAT_MAX

    for k in range(5, max_k, 2):

        opf = ANNUnsupervisedOPF(max_k=k, ann_params=ann)
        clusters = opf.fit_predict(X_val)

        if len(np.bincount(clusters)) > 1:

            score = davies_bouldin_score(X_val, clusters)
            print(f'max_k = {k} - DB-index: {score:.4f}')

            if score < min_score:
                min_score = score
                best_max_k = k
        else:
            break

    print('Chosen max_k on validation set: ', best_max_k)

    return int(best_max_k)


def run_exp(repeat: int, dataset: str, max_k: int, custom_state=None):

    if repeat < 1:
        raise ValueError('You should choose a positive integer as the number of experiment repetitions.')

    # Reset the random seed generator
    np.random.seed(custom_state)

    states = np.random.randint(1e6, size=repeat)

    # Choose dataset function
    load_dataset_func = d.switch(dataset)

    # Load raw dataset
    X, y = load_dataset_func(d.data_path.get(dataset))

    print('Loaded dataset: ', dataset)
    print('Data dimensions: ', X.shape)

    scaler = StandardScaler()

    # Repeat experiments i times...
    for i in range(repeat):

        log = pd.DataFrame(data=None, columns=p.out_columns)

        # Split data into train and validation sets
        idx_train, idx_val = train_test_split(list(range(X.shape[0])), test_size=0.2, random_state=states[i])

        X_train = scaler.fit_transform(X[idx_train])
        X_val = scaler.transform(X[idx_val])

        print('After split: ', X[idx_train].shape, X[idx_val].shape)

        # Guarantees max_k < X_val set size
        max_k = int(np.min([max_k, X_val.shape[0] - 1]))

        # Search for max_k that optimizes OPF clustering structure on validation set
        k = get_max_k(X_val, max_k=max_k, ann_name='knn', custom_state=states[i])

        for ann, prm in p.prm.items():

            row = {'algorithm': f'opf_{ann}'}

            print('Building knn graph with `', ann, '`...')

            # Search for max_k that optimizes OPF clustering structure on validation set
            # k = get_max_k(X_val, max_k=max_k, ann_name=ann, custom_state=states[i])

            ann_prm = prm.copy()

            if ann in ['kdtree', 'hnsw']:
                ann_prm.update({'random_state': states[i]})

            print('Initialize with OPF...')

            tic = time.time()

            opf = ANNUnsupervisedOPF(max_k=k, ann_params=ann_prm)
            clusters = opf.fit_predict(X_train)

            # Number of clusters computed with OPF
            nc = len(np.bincount(clusters))

            # Get centroids ordered by cluster label (0-c)
            # prototypes = None
            prototypes = np.empty(shape=(nc, X_train.shape[-1]))
            for c in range(nc):
                prototypes[c] = [j.features for j in opf.subgraph.nodes if (j.pred == -1 and j.cluster_label == c)][0]

            # Initialize GMM parameters based on OPF's clusters information (weights, means, covariances)
            theta = ge.init_parameters(X_train, clusters,
                                       centroids=prototypes, inv_cov=True, custom_state=states[i])

            row['opf_init_time'] = time.time() - tic

            row['max_k'] = k
            row['k'] = opf.subgraph.best_k
            row['n_clusters'] = nc

            print('Clustering with GMM algorithm...')

            for init_method in ['opf', 'kmeans', 'random']:

                if init_method == 'kmeans':

                    print('Initialize with k-means...')

                    tic = time.time()
                    kmeans = KMeans(nc, init='random', max_iter=500, n_init=5,
                                    algorithm='full', tol=0, random_state=states[i])
                    clusters = kmeans.fit_predict(X_train)

                    prototypes = kmeans.cluster_centers_
                    # prototypes = None

                    # Initialize GMM parameters based on OPF's clusters information (weights, means, covariances)
                    theta = ge.init_parameters(X_train, clusters,
                                               centroids=prototypes, inv_cov=True, custom_state=states[i])

                    row['kmeans_init_time'] = time.time() - tic

                elif init_method == 'random':

                    print('Random initialization...')

                    tic = time.time()
                    theta = ge.random_init_parameters(X_train, n_clusters=nc, inv_cov=True, custom_state=states[i])
                    row['random_init_time'] = time.time() - tic

                # Instantiate Gaussian Mixtures algorithm
                gmm = GaussianMixture(nc, weights_init=theta[0], means_init=theta[1],
                                      precisions_init=theta[2], random_state=states[i])

                # Predict cluster labels on training data
                preds = gmm.fit_predict(X_train, y=y[idx_train])

                # Measures Davies-Bouldin score
                row[f'{init_method}_gmm_db_score'] = davies_bouldin_score(X_train, preds)

                # Measures C-index score
                # row[f'{init_method}_gmm_c_index'] = calc_c_index(X_train, preds)

                # Measures V-measure (with true labels)
                _, _, v_measure = homogeneity_completeness_v_measure(y[idx_train], preds)
                row[f'{init_method}_gmm_v_measure'] = v_measure

                ar_index = adjusted_rand_score(y[idx_train], preds)
                row[f'{init_method}_gmm_adjusted_rand_index'] = ar_index

            log = log.append(row, ignore_index=True)

        f_out = f'./out/{dataset}/results_{get_datetime()}.csv'
        log.to_csv(f_out, mode='w', na_rep='nan')
        print('Saving results to: ', f_out)


if __name__ == '__main__':

    _repeat = int(sys.argv[1])
    _dataset = str(sys.argv[2])
    _max_k = int(sys.argv[3])
    _random_seed = int(sys.argv[4])

    run_exp(_repeat, _dataset, _max_k, custom_state=_random_seed)