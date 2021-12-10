import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.utils.validation import check_random_state

EPSILON = 1e-4


def init_parameters(X, cluster_labels, centroids=None, inv_cov=False, custom_state=None):

    print('Give initial estimations for GMM parameters through OPF clustering...')

    # Guarantee X is a numpy array
    X = np.array(X)

    # Guarantee cluster_labels is a numpy array
    cluster_labels = np.array(cluster_labels)

    n_samples, n_features = X.shape

    # Total number of clusters
    n_clusters = len(np.unique(cluster_labels))

    # Estimated weights based on number of samples per cluster
    weights = np.bincount(cluster_labels) / float(n_samples)

    # Initialize mean vectors
    means = np.empty(shape=(n_clusters, n_features))

    # Initialize covariance matrices
    covariances = np.empty(shape=(n_clusters, n_features, n_features))

    # For each assigned cluster
    for k in range(n_clusters):

        # Query indices from samples assigned to cluster k
        idx_query = np.flatnonzero(cluster_labels == k)

        # Select samples from cluster k
        X_k = X[idx_query]

        means[k] = np.mean(X_k, axis=0) if centroids is None else centroids[k]

        if X_k.shape[0] > 1:
            covariances[k] = np.cov(X_k, rowvar=False)
        else:
            covariances[k] = make_spd_matrix(n_features, random_state=custom_state)

        # Guarantee the covariance matrix has determinant != 0, i.e., it is invertible
        covariances[k].flat[:: n_features + 1] += EPSILON

        if inv_cov:
            # Based on covariance matrices, compute their respective precisions (inverse covariances)
            covariances[k] = np.linalg.inv(covariances[k])

    return weights, means, covariances


def random_init_parameters(X, n_clusters, inv_cov=False, custom_state=None):

    n_samples, n_features = X.shape

    # Estimate initial weights
    weights = np.ones(shape=n_clusters) / float(n_clusters)

    # Estimate initial random mean vector for each component k
    generator = check_random_state(custom_state)
    means = generator.choice(X.flatten(), (n_clusters, n_features))

    # Estimate initial random covariances, which must be a positive semi-defined matrix
    covariances = np.asfarray([
        make_spd_matrix(n_features, random_state=custom_state) for _ in range(n_clusters)])

    for k in range(n_clusters):

        # Guarantee the covariance matrix has determinant != 0, i.e., it is invertible
        covariances[k].flat[:: n_features + 1] += EPSILON

        if inv_cov:
            # Based on covariance matrices, compute their respective precisions (inverse covariances)
            covariances[k] = np.linalg.inv(covariances[k])

    return weights, means, covariances
