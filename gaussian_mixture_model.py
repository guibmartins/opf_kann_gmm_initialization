import time

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

EPSILON = 1e-5
STOP_FLAG = 1e-10
FLOAT_MIN = -1e20
FLOAT_MAX = 1e20


class GaussianMixtureModel:

    def __init__(self, k, init_params=(), iterations=100, custom_state=None):

        self.k = k
        self.weights = None
        self.means = None
        self.covariances = None

        if len(init_params) > 0:
            self.weights, self.means, self.covariances = init_params

        self.iterations = iterations
        self.custom_state = custom_state
        self.fit_log = {}
        self.predict_log = {}

        np.random.seed(custom_state)

    @property
    def k(self):

        return self._k

    @k.setter
    def k(self, k):

        if not isinstance(k, (int, np.int32, np.int64)):
            raise TypeError('')

        if k < 1:
            raise ValueError('')

        self._k = k

    @property
    def weights(self):

        return self._weights

    @weights.setter
    def weights(self, w):

        self._weights = w

    @property
    def means(self):

        return self._means

    @means.setter
    def means(self, means):

        self._means = means

    @property
    def covariances(self):

        return self._covariances

    @covariances.setter
    def covariances(self, covar):

        self._covariances = covar

    @property
    def iterations(self):

        return self._iterations

    @iterations.setter
    def iterations(self, iterations):

        if not isinstance(iterations, (int, np.int32, np.int64)):
            raise TypeError('`Iterations` should be an integer.')

        if iterations < 1:
            raise ValueError('`Iterations` should be a strictly positive integer.')

        self._iterations = iterations

    def _init_parameters(self, X):

        # Estimate initial weights
        weights = np.ones(shape=self.k) / float(self.k)

        # Estimate initial random mean vector for each component k
        means = np.random.choice(X.flatten(), (self.k, X.shape[1]))

        # Estimate initial random covariances, which must be a positive semi-defined matrix
        covariances = np.asfarray([make_spd_matrix(X.shape[1]) for _ in range(self.k)])

        return weights, means, covariances

    def fit(self, X):

        self.fit_log.update({
            'time': time.time(),
            'iter': self.iterations,
            'epsilon': []
        })

        n_samples, n_features = X.shape

        if self.weights is None or \
                self.means is None or \
                self.weights is None:
            # Random initialize parameters
            weights, means, covariances = self._init_parameters(X)
        else:
            # Initialize with pre-defined estimated parameters
            weights = self.weights
            means = self.means
            covariances = self.covariances

        it = 0

        while it < self.iterations:

            likelihood = np.empty(shape=(self.k, n_samples))

            # Expectation (E-step)
            for i in range(self.k):

                # Guarantee the covariance matrix is invertible (Determinant != 0)
                covariances[i].flat[:: n_features + 1] += EPSILON

                # Calculate the likelihood concerning component k for all samples
                likelihood[i] = multivariate_normal(mean=means[i], cov=covariances[i]).pdf(X)

            b = np.empty(shape=(self.k, n_samples))
            log_likelihood = np.empty_like(b)

            for i in range(self.k):

                # Calculate the posterior probabilities for each data sample
                b[i] = (likelihood[i] * weights[i]) / \
                       (np.sum([likelihood[j] * weights[j] for j in range(self.k)], axis=0) + EPSILON)

                # Calculate the expected value of the the log-loss function for theta (t)
                log_likelihood[i] = b[i] * (np.log(weights[i]) +
                                            multivariate_normal(mean=means[i], cov=covariances[i]).logpdf(X))

            self.fit_log.get('epsilon').append(np.sum(np.sum(log_likelihood, axis=0)))
            self.fit_log['iter'] = it + 1

            # Maximization (M-step)
            for i in range(self.k):

                # Update weight
                weights[i] = np.mean(b[i])

                # Update mean
                means[i] = np.sum(b[i].reshape(n_samples, 1) * X, axis=0) / (np.sum(b[i]) + EPSILON)

                # Update covariance
                covariances[i] = np.dot((b[i].reshape(n_samples, 1) *
                                         (X - means[i])).T, (X - means[i])) / (np.sum(b[i]) + EPSILON)

                # Guarantee the covariance matrix is invertible (Determinant != 0)
                covariances[i].flat[:: n_features + 1] += EPSILON

            it += 1

            # Early stopping
            # if len(self.fit_log.get('epsilon')) > 1:
            #
            #     prev_loss = self.fit_log.get('epsilon')[-2]
            #     loss = self.fit_log.get('epsilon')[-1]
            #
            #     if np.abs(loss - prev_loss) < STOP_FLAG:
            #         break

        # Set parameter properties to optimized parameters
        self.weights = weights
        self.means = means
        self.covariances = covariances

        # Set training data duration in ms.
        self.fit_log['time'] = time.time() - self.fit_log.get('time')

    def predict(self, X):

        self.predict_log['time'] = time.time()

        n_samples, n_features = X.shape
        likelihood = np.empty(shape=(self.k, n_samples))

        for i in range(self.k):

            # 'Fixing' singular matrix (adding small values to matrix's main diagonal)
            self.covariances[i].flat[:: n_features + 1] += EPSILON

            # Calculate the likelihood concerning component k for all samples
            likelihood[i] = multivariate_normal(mean=self.means[i], cov=self.covariances[i]).pdf(X)

        bayes_prob = np.empty(shape=(self.k, n_samples))

        for i in range(self.k):
            # Calculate the posterior probabilities for each data sample
            bayes_prob[i] = (likelihood[i] * self.weights[i]) / \
                            (np.sum([likelihood[j] * self.weights[j] for j in range(self.k)], axis=0) + EPSILON)

        # The predicted label is the one with maximum class probability
        preds = np.argmax(bayes_prob, axis=0)

        # Set the testing data duration in ms.
        self.predict_log['time'] = time.time() - self.predict_log.get('time')

        return preds

    def fit_predict(self, X):

        self.fit(X)

        return self.predict(X)
