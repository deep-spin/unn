"""Induce prototypes from a trained model"""

import sys

import torch
from .model import UndirectedNN
from .mnist import BaselineMLP
from .mnist_conv import ConvUNN

import matplotlib.pyplot as plt

def main():

    config = {
        # names of saved models checkpoints
        '1iter_untrained': "vivid-water-11.pt",
        '1iter': "silvery-salad-6.pt",
        '5iter': "firm-shadow-9.pt"
    }

    fname = config[sys.argv[1]]

    torch.set_printoptions(precision=2, sci_mode=False)

    model = torch.load(f"mnist_fwbw_gpu_tanh/{fname}", map_location=torch.device('cpu'))
    model.eval()

    k = model.k
    # k=10
    # I = torch.eye(10)

    # digits = torch.tensor([0, 1, 3, 7], dtype=torch.long)
    digits = torch.arange(10)

    _, axs = plt.subplots(len(digits), k, figsize=(.5*k, .5*len(digits)), constrained_layout=True)

    _, X = model.backward(digits, k=k)


    for i in range(k):
        Xi = X[i]
        Xi = -(1+Xi)/2
        for c in range(len(digits)):
            Xic = Xi[c].view(-1, 28, 28)

            # print("evaluating")
            # print(torch.softmax(model(X), dim=-1))

            ax = axs[c] if k == 1 else axs[c,i]

            ax.imshow(Xic[0], cmap='gray')
            ax.set_xticks(())
            ax.set_yticks(())

    if k == 1:
        axs[0].set_title(f"$t=1$")
    else:
        for i in range(k):
            axs[0, i].set_title(f"$t={i+1}$")

    plt.savefig(f"undirected_convnet_{sys.argv[1]}_full.pdf")
    # plt.show()

if __name__ == '__main__':
    main()
