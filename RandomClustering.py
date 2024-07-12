import random

class RandomClustering:
    def __init__(self, n_clusters=8, affinity="rbf", assign_labels="kmeans"):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.assign_labels = assign_labels
        self.labels_ = None

    def fit(self, X):
        n = X.shape[0]
        c_ids = list(range(self.n_clusters))
        labels = [0] * n
        for i in range(n):
            c_id = random.choice(c_ids)
            labels[i] = c_id
        self.labels_ = labels
        return labels
