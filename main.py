"""Effect of different batchsizes on test loss"""
#!/usr/bin/env python

from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm


# model params
inp, hid1, hid2, out = 784, 392, 50, 10
batch_size = [8, 16, 32, 64, 128, 512, 1024]
epochs = 5
lr = 0.01


class NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(inp, hid1)
        self.h1 = nn.Linear(hid1, hid2)
        self.h2 = nn.Linear(hid2, hid2)
        self.l2 = nn.Linear(hid2, out)

    def forward(self, x):
        x = x.view((-1, 784))
        x = F.relu(self.l1(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return F.softmax(self.l2(x))

nn = NN()
optim = torch.optim.SGD(nn.parameters(), lr=lr)


def train(epoch):
    nn.train()
    print("Training Epoch : {}".format(epoch))
    for data, label in tqdm(train_loader):
        data, label = Variable(data), Variable(label)
        optim.zero_grad()
        output = nn(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optim.step()
    print("Training Loss {:-4f}".format(loss.data[0]))
    return loss.data[0]


def test():
    nn.eval()
    print("Testing..")
    test_loss = 0
    for data, label in tqdm(test_loader):
        data, label = Variable(data), Variable(label)
        output = nn(data)
        test_loss += F.cross_entropy(output, label, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
    test_loss /= len(test_loader.dataset)
    # print("Test Epoch {}, Average Loss {:-4f}".format(epoch, test_loss))
    return test_loss


if __name__ == "__main__":
    tr_batch_loss, ts_batch_loss = OrderedDict(), OrderedDict()
    # load and preprocess MNIST dataset
    transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307, ),
                                                          (0.3081, ))])
    for b_size in batch_size:
        print("Running with batch: {}".format(b_size))
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("/tmp", train=True, download=True,
                           transform=transforms),
            batch_size=b_size)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("/tmp", train=False, download=True,
                           transform=transforms),
            batch_size=b_size)
        tr_loss, ts_loss = [], []
        for e in range(1, epochs + 1):
            tr_loss.append(train(e))
            ts_loss.append(test())
        tr_batch_loss[b_size] = sum(tr_loss)/len(tr_loss)
        ts_batch_loss[b_size] = sum(ts_loss)/len(ts_loss)
    for k, v in tr_batch_loss.items():
        print("{} => {:-4f}".format(k, v))
    tr_items = tr_batch_loss.items()
    ts_items = ts_batch_loss.items()
    x, y_ts = zip(*tr_items)
    x, y_tr = zip(*ts_items)
    plt.plot(x, y_ts, "b", x, y_tr, "g")
    plt.ylabel("Loss")
    plt.xlabel("Mini batch sizes")
    plt.savefig("batchsize.pdf")
    plt.show()