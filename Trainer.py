import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, optim


class Trainer:

    def __init__(self, model, dataset, k=3, lmbda=0.5,):

        self.model = model
        self.lmbda = lmbda
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), betas=(0.9, 0.98), eps=1e-09)

        self.dataset = dataset
        self.k = k

        self.running_loss = 0.0

    def class_loss(self, pred, true_class):
        return self.cross_entropy_loss(pred, true_class)

    def neighbor_loss(self, pred, neighbors):
        pred_log_prob = F.softmax(pred, dim=1)
        neighbor_prob = F.log_softmax(neighbors,  dim=1)
        # print(pred_log_prob)
        # print(neighbor_prob)
        return self.kl_loss(neighbor_prob, pred_log_prob)

    def init_epoch(self):
        self.dataset.init_epoch()
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
        print(loss)
        loss.backward()
        self.optimizer.step()

        self.running_loss += loss.item()

    def epoch(self, epoch):
        self.running_loss = 0.0
        self.init_epoch()
        for i, (inputs, labels) in enumerate(self.dataset):
            if i >= self.k:
                neighbors = self.dataset.query_knn_index(inputs, self.k)
                self.iteration(inputs, neighbors,  labels)
            self.dataset.add_to_knn_index(inputs)

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, self.running_loss / 10))
                self.running_loss = 0.0

    def train(self, epochs=10):
        # self.model.train()
        for epoch in range(epochs):
            self.epoch(epoch)


if __name__ == "__main__":
    from Dataset import Dataset
    from MetaLSTM import MetaLSTM
    from generate import generate_dataset
    import matplotlib.pyplot as plt

    k = 4
    d = 2
    data, labels = generate_dataset(25, d, k, 10, 4, 6)
    # plt.scatter(x=data[:, 0], y=data[:, 1], c=labels)

    meta_lstm = MetaLSTM(input_size=d, hidden_size=32,
                         output_size=k, n_layers=2, batch_size=1)

    dataset = Dataset(data, labels)

    trainer = Trainer(meta_lstm, dataset=dataset)
    trainer.train(2)
    # kl = nn.KLDivLoss(reduce='batchmean')
    # outputs = tensor([[1.0, 2.0, 3.1, 1.02], [24.1, 2, 3, 4]])
    # outputs2 = tensor([[1.0, 2.0, 2, 1.02], [19, 2, 3, 4]])
    # print(F.softmax(outputs, dim=1))
    # print(kl(F.log_softmax(outputs2, dim=1), F.softmax(outputs, dim=1)))
    # loss = nn.CrossEntropyLoss()
    # print(loss(outputs, label))
