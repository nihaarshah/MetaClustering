import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, optim


class Trainer:

    def __init__(self, model, datasets, k=3, lmbda=0.5,):

        self.model = model
        self.lmbda = lmbda
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)

        # self.dataset = dataset
        self.datasets = datasets
        self.k = k

        self.running_loss = 0.0

    def class_loss(self, pred, true_class):
        return self.cross_entropy_loss(pred, true_class)

    def neighbor_loss(self, pred, neighbors):
        pred_log_prob = F.softmax(pred, dim=1)
        neighbor_prob = F.log_softmax(neighbors,  dim=1)

        return self.kl_loss(neighbor_prob, pred_log_prob)

    def init_epoch(self):
        # self.dataset.init_epoch()
        for i in range(len(self.datasets)):
            self.datasets[i].init_epoch()

        self.model.initialize()

    def iteration(self, inputs, neighbors, labels):

        # zero the parameter gradients
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        neighbor_outputs = [self.model(
            tensor(n).reshape(1, -1)) for n in neighbors]

        n_loss = 0.0
        for n_output in neighbor_outputs:
            n_loss += self.neighbor_loss(outputs,
                                         n_output)/len(neighbor_outputs)

        loss = self.lmbda * \
            self.class_loss(outputs, labels) + (1-self.lmbda) * n_loss
        # print(loss)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def epoch(self, epoch):
        epoch_loss = 0.0
        self.init_epoch()
        for d_ind in range(len(self.datasets)):
            dataset_loss = 0.0
            for i, (inputs, labels) in enumerate(self.datasets[d_ind]):
                if i >= self.k:
                    neighbors = self.datasets[d_ind].query_knn_index(
                        inputs, self.k)
                    dataset_loss += self.iteration(inputs, neighbors,  labels)
                self.datasets[d_ind].add_to_knn_index(inputs)

            epoch_loss += dataset_loss/len(self.datasets[d_ind])
        return epoch_loss/len(self.datasets)

    def train(self, epochs=10):
        # self.model.train()
        for epoch in range(epochs):
            print("Epoch %d, loss:%.3f" % (epoch, self.epoch(epoch)))


if __name__ == "__main__":
    from Dataset import Dataset
    from MetaLSTM import MetaLSTM
    from generate import generate_dataset, generate_nonlinear_dataset
    import matplotlib.pyplot as plt

    k = 4
    d = 2
    # data, labels = generate_dataset(25, d, k, 10, 4, 6)
    # data, labels = generate_dataset(25, d, k, 10, 4, 6)
    # plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)

    meta_lstm = MetaLSTM(input_size=d, hidden_size=100,
                         output_size=k, n_layers=2, batch_size=1)

    gauss_datasets = [Dataset(
        *generate_dataset(n=25, d=2, k=4, rng=10, alpha=1, beta=2))
        for _ in range(5)]

    nonlinear_datasets = [Dataset(
        *generate_nonlinear_dataset(n=25, d=2, k=4, rng=10,
                                    alpha=10, beta=2,
                                    a_nl=12, b_nl=2
                                    ))
                          for _ in range(5)]

    trainer = Trainer(meta_lstm, datasets=gauss_datasets+nonlinear_datasets)
    trainer.train(100)
    # kl = nn.KLDivLoss(reduce='batchmean')
    # outputs = tensor([[1.0, 2.0, 3.1, 1.02], [24.1, 2, 3, 4]])
    # outputs2 = tensor([[1.0, 2.0, 2, 1.02], [19, 2, 3, 4]])
    # print(F.softmax(outputs, dim=1))
    # print(kl(F.log_softmax(outputs2, dim=1), F.softmax(outputs, dim=1)))
    # loss = nn.CrossEntropyLoss()
    # print(loss(outputs, label))
