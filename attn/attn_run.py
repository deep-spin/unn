"""prototyping self-attention as an UNN"""
import sys, random

import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# special indices
PAD = 0
MASK = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVE_OUTPUTS = False

def write_line_to_file(filename, *text):
    text = ' '.join(map(str, text))
    print(text)
    with open(filename, 'a', encoding="utf-8") as out:
        out.write(text)
        out.write('\n')


class SelfAttn(torch.nn.Module):

    def __init__(self, vocab_size, d, max_len):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, d)
        # self.emb.weight.requires_grad_(False)
        # self.emb.weight[MASK] = 0

        self.Wk = torch.nn.Linear(d, d, bias=False)
        self.Wv = torch.nn.Linear(d, d, bias=False)
        self.Wq = torch.nn.Linear(d, d, bias=False)
        self.sqd = np.sqrt(d)

        self.PE = make_positional(d, max_len).to(device=device)

    def forward(self, ix):

        n = ix.shape[-1]
        X = self.emb(ix)

        mask = (ix == MASK).unsqueeze(-1)
        X = X.masked_fill(mask, 0)

        X_plus_pos = X + self.PE[:n].unsqueeze(dim=0)
        K = self.Wk(X_plus_pos)
        V = self.Wv(X_plus_pos)
        Q = self.Wq(X_plus_pos)
        S = torch.softmax(Q @ K.transpose(-2, -1) / self.sqd, dim=-1)
        # print(S.shape)
        H = S @ V
        logits = H[mask.squeeze(-1)] @ self.emb.weight.T
        return logits


class UndirectedAttn(SelfAttn):

    def __init__(self, vocab_size, d, max_len, n_iter, order='default'):
        super().__init__(vocab_size, d, max_len)

        self.n_iter = n_iter
        self.order = order
        self.d = d

        self._S = [] #torch.FloatTensor(max_len, max_len)
        self._S_next = [] #torch.FloatTensor(max_len, max_len)
        self._logits = [] #torch.FloatTensor(max_len, max_len)


    def _update_S(self, Q, K, H, V, pad):
        QK = Q @ K.transpose(-2, -1)
        HV = H @ V.transpose(-2, -1)

        pad = pad.to(dtype=Q.dtype)
        mask = (pad + pad.transpose(-2, -1)) > 0
        mask_inf = (torch.where(mask, -9999, 0)
            .to(device=Q.device, dtype=Q.dtype))

        S = torch.softmax((QK + HV + mask_inf) / self.sqd, dim=-1)
        #S = torch.softmax((QK + HV) / self.sqd, dim=-1)
        S = S.masked_fill(mask, 0.0)
        return S

    def _update_Q(self, X_plus_pos, S, K):
        return self.Wq(X_plus_pos) + S @ K

    def _update_K(self, X_plus_pos, S, Q):
        return self.Wk(X_plus_pos) + S.transpose(-2, -1) @ Q

    def _update_V(self, X_plus_pos, S, H):
        return self.Wv(X_plus_pos) + S.transpose(-2, -1) @ H

    def _update_H(self, S, V):
        return S @ V

    def _init_vars(self, batch_size, n, d):
        Q = torch.zeros(batch_size, n, d).to(device=device)
        K = torch.zeros(batch_size, n, d).to(device=device)
        V = torch.zeros(batch_size, n, d).to(device=device)
        S = torch.zeros(batch_size, n, n).to(device=device)
        H = torch.zeros(batch_size, n, d).to(device=device)
        return Q, K, V, S, H

    def forward(self, ix):
        self._S = [] 
        self._S_next = [] 
        self._logits = [] 
        if self.order == 'default':
            return self._forward_default(ix)
        elif self.order == 'random':
            return self._forward_random(ix)

    def _forward_default(self, ix):
        n_iter = self.n_iter

        n = ix.shape[-1]
        X = self.emb(ix)

        mask = (ix == MASK).unsqueeze(-1)
        pad = (ix == PAD).unsqueeze(-1)
        X = X.masked_fill(mask, 0)

        Q, K, V, S, H = self._init_vars(X.shape[0], n, self.d)

        if n_iter <= 0:
            if self.training:
                # Pick a random number of updates between 1 and 5 in training mode
                n_iter = random.randint(1, 5)
            else:
                # Make 3 updates in eval mode or as specified
                n_iter = 3

        for _ in range(n_iter):
            X_plus_pos = X + self.PE[:n].unsqueeze(0)

            # "forward pass"
            Q = self._update_Q(X_plus_pos, S, K)
            K = self._update_K(X_plus_pos, S, Q)
            V = self._update_V(X_plus_pos, S, H)

            # Q,K,V are [B x len x hid]
            # S is [B x len x len]
            S = self._update_S(Q, K, H, V, pad)
            H = self._update_H(S, V)

            self._S.append(S)

            # "backward pass"

            S = self._update_S(Q, K, H, V, pad)
            V = self._update_V(X_plus_pos, S, H)
            K = self._update_K(X_plus_pos, S, Q)
            Q = self._update_Q(X_plus_pos, S, K)

            # logits for masked
            X_mask = (
                  V[mask.squeeze(-1)] @ self.Wv.weight +
                  K[mask.squeeze(-1)] @ self.Wk.weight +
                  Q[mask.squeeze(-1)] @ self.Wq.weight 
            )
            logits = X_mask @ self.emb.weight.T

            X = torch.masked_scatter(X, mask, X_mask)

            # Saving these in the model for reporting purposes
            self._S_next.append(S)
            self._logits.append(logits)

        return logits


    def _forward_random(self, ix):
        n_iter = self.n_iter

        n = ix.shape[-1]
        X = self.emb(ix)

        self.Q, self.K, self.V, self.S, self.H = \
            self._init_vars(X.shape[0], n, self.d)

        mask = (ix == MASK).unsqueeze(-1)
        pad = (ix == PAD).unsqueeze(-1)
        X = X.masked_fill(mask, 0)

        if n_iter <= 0:
            if self.training:
                # Pick a random number of updates between 1 and 5 in training mode
                n_iter = random.randint(1, 5)
            else:
                # Make 3 updates in eval mode or as specified
                n_iter = 3
                if n_iter < 0:
                    n_iter = -n_iter

        for _ in range(n_iter):
            X_plus_pos = X + self.PE[:n].unsqueeze(0)

            # START RANDOM
            ops = [
                "self.Q = self._update_Q(X_plus_pos, self.S, self.K)",
                "self.K = self._update_K(X_plus_pos, self.S, self.Q)",
                "self.V = self._update_V(X_plus_pos, self.S, self.H)",
                "self.S = self._update_S(self.Q, self.K, self.H, self.V, pad)",
                "self.H = self._update_H(self.S, self.V)",
            ]

            random.shuffle(ops)
            for op in ops:
                exec(op)
            # END RANDOM

            # logits for masked
            X_mask = (
                  self.V[mask.squeeze(-1)] @ self.Wv.weight +
                  self.K[mask.squeeze(-1)] @ self.Wk.weight +
                  self.Q[mask.squeeze(-1)] @ self.Wq.weight
            )
            logits = X_mask @ self.emb.weight.T

            X = torch.masked_scatter(X, mask, X_mask)

            # Saving these in the model for reporting purposes
            self._S.append(self.S)
            self._S_next.append(self.S)
            self._logits.append(logits)

        return logits


def make_positional(d, max_len):
    ix = torch.arange(max_len)
    di = torch.arange(d // 2).to(dtype=torch.float32) / (d // 2)
    Psin = torch.sin((10000 ** -di).unsqueeze(0) * ix.unsqueeze(1))
    Pcos = torch.cos((10000 ** -di).unsqueeze(0) * ix.unsqueeze(1))
    return torch.column_stack([Psin, Pcos])


def main(dataset, k, order='default'):
    # device = 'cuda'

    torch.manual_seed(2)
    random.seed(2)  # for the python random.choice
    max_iter = 150 if dataset == 'sequences.pt' else 1000
    batch_size = 256
    d = 256
    learning_rate = 0.0001 #if dataset == 'sequences.pt' else 0.0001
    early_stopping_epochs = 100 if dataset == 'sequences.pt' else 100

    run = wandb.init(entity='unn', project=dataset[:-3],
            config={'dataset': dataset, 'unn_iter': k, 'batch_size': batch_size,
                    'dim': d, 'max_iter': max_iter, 'lr': learning_rate,
                    'order': order})


    # dataset = "words.pt"
    # dataset = "sequences.pt"
    data_train, data_valid = torch.load(dataset)

    # annoying aside because i constructed the two datasets differently
    tensors = getattr(data_train, 'tensors', None)
    if tensors is None:
        tensors = data_train.dataset.tensors

    X_train, _ = tensors

    vocab_size = X_train.max() + 1
    max_len = X_train.shape[-1]

    sa = UndirectedAttn(vocab_size, d, max_len, n_iter=k, order=order)
    # sa = SelfAttn(vocab_size, d, max_len)
    print(sa)
    if device == 'cuda':
        #print('is cuda')
        sa.cuda()
        for param in sa.parameters():
            param = param.to(device='cuda')

    wandb.watch(sa, log_freq=1)
    opt = torch.optim.Adam(sa.parameters(), lr=learning_rate)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size, shuffle=False)

    print(f"N training {len(data_train)} n test {len(data_valid)}")

    def validate():
        correct = 0
        total = 0
        logged = 0
        max_sent_len = 10
        min_sent_len = 20

        if SAVE_OUTPUTS and evals >= max_iter-1:
            filename = f"log/{dataset[:-3]}-{wandb.run.name}.txt"
            write_line_to_file(filename, f"Step: {evals}")

        for batch in valid_loader:
            sa.eval()
            X, y = batch

            nonzeros = torch.nonzero(X).shape[0]
            if SAVE_OUTPUTS and nonzeros < min_sent_len:
                # Save X and the attention weights to a file
                continue

            # print('X:', X, torch.nonzero(X).shape[0])
            X, y = (x.to(device=device) for x in (X, y))
            #print('X,y', X, y)
            with torch.no_grad():
                tgt_logits = sa(X)
            tgt_pred  = torch.argmax(tgt_logits, dim=-1)
            tgt_ix = y[y > 0]
            correct += (tgt_pred == tgt_ix).sum().item()
            total += tgt_ix.shape[0]

            if SAVE_OUTPUTS and evals >= max_iter:
                if nonzeros >= min_sent_len:
                    logged += 1
                    # Save to a file X and the attention weights
                    write_line_to_file(filename, f"X({nonzeros}): {X}")
                    write_line_to_file(filename, f"S: {sa._S}")
                    write_line_to_file(filename, f"S_next: {sa._S_next}")
                    write_line_to_file(filename, f"logits: {sa._logits}")

                if logged >= 10:
                    write_line_to_file(filename, f"==========================")
                    break

            # break # only process one element

        accuracy = 0 if total == 0 else correct/total
        print(f"accuracy {accuracy}")
        return accuracy

    n_updates = 0
    evals = 0
    break_loop = False

    val_acc = validate()
    max_val_acc = val_acc
    max_val_epoch = 0
    for i in range(max_iter):
        if break_loop:
            break

        # training
        for batch in train_loader:
            sa.train()
            opt.zero_grad()
            X, y = batch
            X, y = (x.to(device=device) for x in (X, y))
            #print(sa, X, y)
            tgt_logits = sa(X)
            tgt_ix = y[y > 0]
            loss = F.cross_entropy(tgt_logits, tgt_ix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sa.parameters(), 10)
            opt.step()
            n_updates += 1

            if n_updates % 1000 == 0:
                evals += 1
                print(f"{n_updates} updates; loss {loss.item()}")
                val_acc = validate()
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    max_val_epoch = evals

                    # Saving best model
                    model_path = f"saved_models/{dataset[:-3]}-{wandb.run.name}.pt"
                    torch.save(sa, model_path)


                train_loss = loss.item()
                wandb.log({
                    'val_acc': val_acc,
                    'loss': train_loss,
                    'max_val_acc': max_val_acc,
                    'max_val_epoch': max_val_epoch,
                    'n_updates': n_updates,
                    'S': sa._S,
                    'S_next': sa._S_next,
                    'logits': sa._logits
                })

                # Early stopping
                if early_stopping_epochs < evals - max_val_epoch:
                    print('Early stopping', early_stopping_epochs, evals, max_val_epoch)
                    break_loop = True
                    break

                # Max iterations reached
                if evals >= max_iter:
                    print('Max iters reached.')
                    break_loop = True
                    break

if __name__ == '__main__':
    _, dataset, k, order = sys.argv
    main(dataset, int(k), order)

