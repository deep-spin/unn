import math
from functools import partial

from dataclasses import dataclass
from typing import Optional
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import wandb


@dataclass
class MNISTConvConfigSchema:
    project: str = 'mnist'
    entity: str = 'unn'
    baseline: bool = False
    dropout: float = 0.3
    seed: int = 42
    lr: float = 0.001
    batch_size: int = 512
    epochs: int = 20
    unn_iter: int = 1
    unn_order: str = "fw"
    unn_y_init: str = "zero"
    backward_loss_coef: float = 0
    constrained: bool = True # False means we want *unn_iter* number of layers but without shared weights
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_config(show=False):
    conf = OmegaConf.from_cli()

    # validate against schema
    schema = OmegaConf.structured(MNISTConvConfigSchema)
    conf = OmegaConf.merge(schema, conf)

    if show:
        print(OmegaConf.to_yaml(conf))

    conf = OmegaConf.to_container(conf)

    return conf


class ConvUNN(torch.nn.Module):

    def __init__(self, k, n_classes, dropout_p=0, unn_order='fw', y_init="zero", constrained=True):
        super().__init__()
        self.k = k
        self.n_classes = n_classes
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.unn_order = unn_order
        self.y_init = y_init
        self.constrained = constrained

        # fixed arch. lazy
        d0 = 28
        d1 = 6
        n1 = 32
        d2 = 4
        n2 = 64
        d3 = 5
        n3 = n_classes

        self.stride = 2

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.d0 = d0
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

        if not self.constrained:
            # n1 = int(n1/self.k)
            n2 = int(n2/self.k)
            # self.n1 = n1
            self.n2 = n2
            print('self.n1,n2', self.n1, self.n2)

        self.h1_dim = (self.d0 - self.d1) // self.stride + 1
        self.h2_dim = (self.h1_dim - self.d2) // self.stride + 1

        self.W1 = torch.nn.Parameter(torch.empty(n1, 1, d1, d1))
        self.b1 = torch.nn.Parameter(torch.empty(n1))

        self.W2 = torch.nn.Parameter(torch.empty(n2, n1, d2, d2))
        self.b2 = torch.nn.Parameter(torch.empty(n2))

        self.W3 = torch.nn.Parameter(torch.empty(n3, n2 * self.h2_dim * self.h2_dim))
        self.b3 = torch.nn.Parameter(torch.empty(n3))

        print("self.W1", self.W1.shape)
        print("self.b1", self.b1.shape)
        print("self.W2", self.W2.shape)
        print("self.b2", self.b2.shape)
        print("self.W3", self.W3.shape)
        print("self.b3", self.b3.shape)

        torch.nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
                                       # nonlinearity='tanh')
        torch.nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
                                       # nonlinearity='tanh')
        torch.nn.init.kaiming_uniform_(self.W3, a=math.sqrt(5))

        fan_in_1, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W1)
        bound_1 = 1 / math.sqrt(fan_in_1)
        torch.nn.init.uniform_(self.b1, -bound_1, bound_1)

        fan_in_2, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W2)
        bound_2 = 1 / math.sqrt(fan_in_2)
        torch.nn.init.uniform_(self.b2, -bound_2, bound_2)

        fan_in_3, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W3)
        bound_3 = 1 / math.sqrt(fan_in_3)
        torch.nn.init.uniform_(self.b3, -bound_3, bound_3)

        # print('constrained', constrained)
        if not constrained:
            for j in range(self.k - 1):
                # print('j', j)
                i = j+1
                setattr(self, f"W1_{i}", torch.nn.Parameter(torch.empty(n1, 1, d1, d1)))
                setattr(self, f"b1_{i}", torch.nn.Parameter(torch.empty(n1)))

                setattr(self, f"W2_{i}", torch.nn.Parameter(torch.empty(n2, n1, d2, d2)))
                setattr(self, f"b2_{i}", torch.nn.Parameter(torch.empty(n2)))

                setattr(self, f"W3_{i}", torch.nn.Parameter(torch.empty(n3, n2 * self.h2_dim * self.h2_dim)))
                setattr(self, f"b3_{i}", torch.nn.Parameter(torch.empty(n3)))

                W1_temp = getattr(self, f"W1_{i}")
                b1_temp = getattr(self, f"b1_{i}")
                W2_temp = getattr(self, f"W2_{i}")
                b2_temp = getattr(self, f"b2_{i}")
                W3_temp = getattr(self, f"W3_{i}")
                b3_temp = getattr(self, f"b3_{i}")

                print("W1_temp", W1_temp.shape)
                print("b1_temp", b1_temp.shape)
                print("W2_temp", W2_temp.shape)
                print("b2_temp", b2_temp.shape)
                print("W3_temp", W3_temp.shape)
                print("b3_temp", b3_temp.shape)

                torch.nn.init.kaiming_uniform_(W1_temp, a=math.sqrt(5))
                                               # nonlinearity='tanh')
                torch.nn.init.kaiming_uniform_(W2_temp, a=math.sqrt(5))
                                               # nonlinearity='tanh')
                torch.nn.init.kaiming_uniform_(W3_temp, a=math.sqrt(5))

                fan_in_1, _ = torch.nn.init._calculate_fan_in_and_fan_out(W1_temp)
                bound_1 = 1 / math.sqrt(fan_in_1)
                torch.nn.init.uniform_(b1_temp, -bound_1, bound_1)

                fan_in_2, _ = torch.nn.init._calculate_fan_in_and_fan_out(W2_temp)
                bound_2 = 1 / math.sqrt(fan_in_2)
                torch.nn.init.uniform_(b2_temp, -bound_2, bound_2)

                fan_in_3, _ = torch.nn.init._calculate_fan_in_and_fan_out(W3_temp)
                bound_3 = 1 / math.sqrt(fan_in_3)
                torch.nn.init.uniform_(b3_temp, -bound_3, bound_3)


    def _update_X(self, H1):
        return torch.conv_transpose2d(H1, weight=self.W1, stride=self.stride)

    def _update_H1(self, X, H2):
        H1_fwd = torch.conv2d(X, self.W1, self.b1, stride=self.stride)
        H1_bwd = torch.conv_transpose2d(H2, weight=self.W2, stride=self.stride)
        return H1_fwd + H1_bwd

    def _update_H2(self, H1, Y):
        h1_dim = (self.d0 - self.d1) // self.stride + 1
        h2_dim = (h1_dim - self.d2) // self.stride + 1
        H2_fwd = torch.conv2d(H1, self.W2, self.b2, stride=self.stride)
        H2_bwd = (Y @ self.W3).reshape(-1, self.n2, h2_dim, h2_dim)
        return H2_fwd + H2_bwd

    def _update_Y(self, H2):
        # flatten
        H2_ = H2.view(H2.shape[0], -1)
        return H2_ @ self.W3.T + self.b3


    def _update_H1_unconstrained(self, X, H2, i):
        W1 = getattr(self, f"W1_{i}")
        b1 = getattr(self, f"b1_{i}")
        W2 = getattr(self, f"W2_{i}")
        H1_fwd = torch.conv2d(X, W1, b1, stride=self.stride)
        H1_bwd = torch.conv_transpose2d(H2, weight=W2, stride=self.stride)
        return H1_fwd + H1_bwd

    def _update_H2_unconstrained(self, H1, Y, i):
        W2 = getattr(self, f"W2_{i}")
        b2 = getattr(self, f"b2_{i}")
        W3 = getattr(self, f"W3_{i}")
        h1_dim = (self.d0 - self.d1) // self.stride + 1
        h2_dim = (h1_dim - self.d2) // self.stride + 1
        H2_fwd = torch.conv2d(H1, W2, b2, stride=self.stride)
        H2_bwd = (Y @ W3).reshape(-1, self.n2, h2_dim, h2_dim)
        return H2_fwd + H2_bwd

    def _update_Y_unconstrained(self, H2, i):
        # flatten
        W3 = getattr(self, f"W3_{i}")
        b3 = getattr(self, f"b3_{i}")
        H2_ = H2.view(H2.shape[0], -1)
        return H2_ @ W3.T + b3

    def forward(self, X):

        h1_dim = (self.d0 - self.d1) // self.stride + 1
        h2_dim = (h1_dim - self.d2) // self.stride + 1

        b = X.shape[0]
        H2 = torch.zeros((b, self.n2, h2_dim, h2_dim), device=X.device)
        
        # Initialize Y according to setup
        if self.y_init == "zero":
            # Initialize Y with zeros by default
            Y = torch.zeros(b, self.n3, device=X.device) 
        elif self.y_init == "rand":
            # Initialize Y as a random probability distribution
            Y = torch.rand(b, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)
        elif self.y_init == "uniform":
            # Initialize Y as a random probability distribution
            Y = torch.zeros(b, self.n3, device=X.device) 
            Y = torch.softmax(Y, dim=-1)

        mask_H1 = torch.ones(b, self.n1, h1_dim, h1_dim, device=X.device)
        mask_H1 = self.dropout(mask_H1)

        mask_H2 = torch.ones(b, self.n2, h2_dim, h2_dim, device=X.device)
        mask_H2 = self.dropout(mask_H2)

        for i in range(self.k):
            if self.constrained or i==0: # Standard training of UNN with k steps of coordinate descent
                # for _ in range(self.k):
                H1 = self._update_H1(X, H2)
                H1 = torch.tanh(H1)
                H1 = H1 * mask_H1

                H2 = self._update_H2(H1, Y)
                H2 = torch.tanh(H2)
                H2 = H2 * mask_H2

                Y_logits = self._update_Y(H2)
                Y = torch.softmax(Y_logits, dim=-1)

                if self.unn_order == 'fb':
                    H2 = self._update_H2(H1, Y)
                    H2 = torch.tanh(H2)
                    H2 = H2 * mask_H2
            else: # Model with k layers but without shared wieghts
                H1 = self._update_H1_unconstrained(X, H2, i)
                H1 = torch.tanh(H1)
                H1 = H1 * mask_H1

                H2 = self._update_H2_unconstrained(H1, Y, i)
                H2 = torch.tanh(H2)
                H2 = H2 * mask_H2

                Y_logits = self._update_Y_unconstrained(H2, i)
                Y = torch.softmax(Y_logits, dim=-1)

                if self.unn_order == 'fb':
                    H2 = self._update_H2(H1, Y)
                    H2 = torch.tanh(H2)
                    H2 = H2 * mask_H2

        return Y_logits

    def backward(self, y, k=None, return_all_x=True):

        if k is None:
            k = self.k

        Y = torch.nn.functional.one_hot(y, num_classes=self.n_classes)
        b = Y.shape[0]

        all_X = []

        H1 = torch.zeros(b, self.n1, self.h1_dim, self.h1_dim, device=y.device)

        X = torch.zeros(b, 1, self.d0, self.d0, device=y.device)
        Y = Y.to(dtype=X.dtype)

        for i in range(k):
            H2 = self._update_H2(H1, Y)
            H2 = torch.tanh(H2)
            H1 = self._update_H1(X, H2)
            H1 = torch.tanh(H1)
            Xp = self._update_X(H1)
            X = torch.tanh(Xp)

            if return_all_x:
                all_X.append(X.detach().clone())

            if self.unn_order == 'fb':
                H1 = self._update_H1(X, H2)
                H1 = torch.tanh(H1)

        # returns pre-activation (logit) X as well as matrix of all Xs.
        return Xp, all_X


class BaselineConv(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=6, stride=2),
            torch.nn.Tanh(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.Tanh(),
            torch.nn.Flatten(-3, -1),
            torch.nn.Linear(in_features=64*5*5, out_features=n_classes)
        )

    def forward(self, x):
        output = self.net(x)
        return output


def main():

    conf = load_config(show=True)
    run = wandb.init(project=conf['project'],
                     entity=conf['entity'],
                     config=conf)

    # Step 1. Load Dataset
    train_and_dev_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

    n_train_and_dev = len(train_and_dev_dataset)
    n_dev = 10000
    train_dataset, dev_dataset = random_split(
        train_and_dev_dataset,
        [n_train_and_dev - n_dev, n_dev],
        generator=torch.Generator().manual_seed(42)
    )

    print("Train data", len(train_dataset))
    print("Dev   data", len(dev_dataset))
    print("Test  data", len(test_dataset))

    torch.manual_seed(conf['seed'])

    batch_size = conf['batch_size']
    _, w, h = train_dataset[0][0].shape
    input_dim = w * h
    output_dim = 10

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if conf['baseline']:
        model = BaselineConv(input_dim, conf['hidden_dim'], output_dim)
    else:
        model = ConvUNN(k=conf['unn_iter'],
                        n_classes=output_dim,
                        dropout_p=conf['dropout'],
                        unn_order=conf['unn_order'],
                        y_init=conf['unn_y_init'],
                        constrained=conf['constrained'])

    if conf['device'] == 'cuda':
        model = model.cuda()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])

    print('Num parameters:', sum(p.numel() for p in model.parameters()))

    # for p in model.parameters():
    #     print(p.numel())

    for name, param in model.named_parameters():
        print(name, param.numel())

    train_model(model,
                train_loader,
                dev_loader,
                test_loader,
                optimizer,
                conf,
                partial(get_x_y, cuda=conf['device'] == 'cuda'))




def get_x_y(batch, cuda):
    # *_, w, h = batch[0].shape
    # images = batch[0].view(-1, w * h)  # between [0, 1].
    images = batch[0]
    images = 2*images - 1  # [between -1 and 1]
    labels = batch[1]

    if cuda:
        images = images.cuda()
        labels = labels.cuda()

    return images, labels


def train_model(model, train_loader, dev_loader, test_loader,
                optimizer, conf, get_x_y):

    n_train = len(train_loader.dataset)

    best_dev_acc = 0
    best_dev_acc_test_acc = None
    best_dev_acc_epoch = None

    # accuracy_dev = eval_model(model, dev_loader, get_x_y)
    # accuracy_test = eval_model(model, test_loader, get_x_y)
    # print("Before training acc", accuracy_dev, accuracy_test)

    # computes softmax and then the cross entropy
    loss_fw = torch.nn.CrossEntropyLoss(reduction='none')
    loss_bw = torch.nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(conf['epochs']):
        loss_fw_train = 0
        loss_bw_train = 0
        accuracy_train = 0

        for batch_id, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x, y = get_x_y(batch)

            # [batch x n_classes]
            logits_fw = model(x)
            # [batch]
            loss_val_fw = loss_fw(logits_fw, y)

            loss_avg = loss_val_fw.mean()
            if conf['backward_loss_coef'] > 0:

                # [batch x 1 x 28 x 28]
                logits_bw, _ = model.backward(y)
                # [batch x 1 x 28 x 28]
                loss_val_bw = loss_bw(logits_bw, (x>0).to(dtype=x.dtype))

                loss_avg = loss_avg + conf['backward_loss_coef'] * loss_val_bw.mean()

            loss_avg.backward()
            optimizer.step()

            loss_fw_train += loss_val_fw.sum().item()
            if conf['backward_loss_coef'] > 0:
                loss_bw_train += loss_val_bw.mean(dim=-1).sum().item()
            accuracy_train += (logits_fw.argmax(dim=1) == y).sum().item()

        accuracy_dev = eval_model(model, dev_loader, get_x_y)
        accuracy_test = eval_model(model, test_loader, get_x_y)

        loss_fw_train /= n_train  # average sample loss
        loss_bw_train /= n_train  # average sample loss
        accuracy_train /= n_train

        if accuracy_dev > best_dev_acc:
            best_dev_acc = accuracy_dev
            best_dev_acc_test_acc = accuracy_test
            best_dev_acc_epoch = epoch
        torch.save(model, f'{wandb.run.name}.pt')

        log = {
            'epoch': epoch,
            'loss_fw_train': loss_fw_train,
            'loss_bw_train': loss_bw_train,
            'acc_train': accuracy_train,
            'acc_dev': accuracy_dev,
            'acc_test': accuracy_test
        }

        wandb.log(log)
        print(log)


def eval_model(model, test_loader, get_x_y):
    correct = 0
    total = len(test_loader.dataset)
    model.eval()
    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):

            x, y = get_x_y(batch)
            outputs = model(x)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()

    accuracy = correct/total
    return accuracy


if __name__ == '__main__':
    main()
