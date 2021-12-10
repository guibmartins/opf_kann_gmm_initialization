# Approximated Nearest Neighbors Search - Algorithms' parameters
nn_kdtree = {'name': 'kdtree', 'leaf_size': 30, 'random_state': None, 'n_jobs': -1}
nn_annoy = {'name': 'annoy', 'distance': 'euclidean', 'n_trees': 15, 'n_jobs': -1}
nn_hnsw = {'name': 'hnsw', 'distance': 'l2', 'ef': 10, 'M': 100, 'random_state': None, 'n_jobs': -1}
knn = {'name': 'knn', 'n_neighbors': 1, 'algorithm': 'brute', 'distance': 'minkowski', 'p': 2, 'n_jobs': -1}

prm = {'knn': knn, 'kdtree': nn_kdtree, 'annoy': nn_annoy, 'hnsw': nn_hnsw}

out_columns = ['algorithm', 'max_k', 'k', 'n_clusters']
