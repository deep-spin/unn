"""prototyping self-attention as an UNN"""
import sys, random

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attn_run import UndirectedAttn
from show_attn import show_plots


# special indices
PAD = 0
MASK = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def incorrect_digits(X, logits):
    idx = 0
    previous_digit = -1
    for x_i in X:
      if x_i == 1:
        log_i = logits[idx]
        print(f"[{log_i.argmax()}]")
        idx += 1
      else:
        print(f" {x_i}")



if __name__ == '__main__':
    # _, model_path = sys.argv
    dataset = 'sequences.pt'
    model_path = f"saved_models/245-default-k=1.pt"
    # model_path = f"saved_models/247-default-k=2.pt"
    # model_path = f"saved_models/246-default-k=-1.pt"
    # model_path = f"saved_models/247-default-k=4.pt"
    # model_path = f"saved_models/251-random-k=4.pt"
    # model_path = f"saved_models/252-random-k=1.pt"
    model = torch.load(model_path, map_location=torch.device(device))
    # model = model.to(device=device)
    print(model)
    n_iter = model.n_iter
    order = 'default' if not hasattr(model, 'order') else model.order
    print(n_iter, order)

    if not hasattr(model, 'order'):
        model.order = 'default'

    _, data_valid = torch.load(dataset)
    valid_loader = DataLoader(data_valid, batch_size=1, shuffle=False)

    for batch in valid_loader:
        model.eval()
        X, y = batch
        X, y = (x.to(device=device) for x in (X, y))

        nonzero_X = torch.nonzero(X).shape[0]
        nonzero_y = torch.nonzero(y).shape[0]
        print('X,y,nzX,nzY', X, y, nonzero_X, nonzero_y, y[:nonzero_y], X[0][0])

        if nonzero_y < 3 or X[0][0] != 29:
            print('skipping...')
            continue

        with torch.no_grad():
            tgt_logits = model(X)

        print('pred:', torch.argmax(tgt_logits, -1), y[0][:nonzero_y])

        if not torch.allclose(torch.argmax(tgt_logits, -1), y[0][:nonzero_y]):
            print('Wrong prediction')
            # continue
        else:
            print('Correct prediction')
            

        S = model._S
        S_next = model._S_next

        show_plots(X, S, S_next)

        # break
