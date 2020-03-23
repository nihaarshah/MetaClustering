import numpy as np
import hnswlib
from torch import tensor


class Dataset:

    def __init__(self, data, labels, batch_size=1, shuffle=True):

        self.data = tensor(data).float()
        self.labels = tensor(labels).long()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = data.shape[1]
        self.index = np.arange(len(self.labels))

        self.max_elements = len(data)

        self.init_epoch()

    def init_epoch(self):
        self._shuffle_data()
        self._init_knn_index()

    def _shuffle_data(self):
        np.random.shuffle(self.index)

    def _init_knn_index(self):
        self.knn_index = hnswlib.Index(space='l2', dim=self.dim)
        self.knn_index.init_index(
            max_elements=self.max_elements, ef_construction=100, M=16)
        self.knn_index.set_ef(10)
        self.knn_index.set_num_threads(4)

    def add_to_knn_index(self, x):
        self.knn_index.add_items(x.reshape(-1, self.dim))

    def query_knn_index(self, x, k=1):
        ids, dist = self.knn_index.knn_query(x.reshape(-1, self.dim), k)
        return self.knn_index.get_items(ids.reshape(-1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, ind):

        _ind = self.index[ind]
        X = self.data[_ind, :].reshape(-1, self.dim)
        y = self.labels[_ind].reshape(-1)

        return X, y


if __name__ == "__main__":

    data = np.random.rand(20, 2)
    labels = np.random.rand(20, 1).round().astype(int)
    D = Dataset(data, labels)
    G = D.generator()
    D.add_to_knn_index(data[0:10, :])
    for i in range(20):
        x, y = next(G)
        # D._add_to_knn_index(x)
    # print(x)
    # print(D.query_knn_index(x, 3))
    #print(D.knn_index.knn_query(x, 3))
