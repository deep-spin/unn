# generate sequence data and save to file

from itertools import combinations

import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

PAD = 0
MASK = 1

def make_all_sequences(min_length, max_length, n_vocab, ratio_missing=.1):

    seqs = []
    y = []

    for n in range(min_length, max_length):

        # make all sequences of length n
        max_start = n_vocab - n + 1
        pad = torch.full(size=(max_length - n - 1,), fill_value=PAD, dtype=torch.long)

        max_n_missing = 1 + int(np.ceil(ratio_missing * n))

        for start in range(0, max_start):
            seq = 2 + torch.arange(start, start+n)

            for n_missing in range(1, max_n_missing):
                for ix in combinations(np.arange(n), n_missing):
                    ix = np.array(ix)

                    seq_fwd = seq.clone()
                    seq_rev = seq.clone().flip(-1)

                    y_fwd = seq_fwd[ix]
                    seq_fwd[ix] = MASK

                    y_rev = seq_rev[ix]
                    seq_rev[ix] = MASK

                    seqs.extend([seq_fwd, seq_rev])
                    y.extend([y_fwd, y_rev])

    X = pad_sequence(seqs, batch_first=True, padding_value=PAD)
    y = pad_sequence(y, batch_first=True, padding_value=PAD)

    raw_dataset = data.TensorDataset(X, y)

    N = len(seqs)
    n_test = int(.1 * N)
    lengths = [N - n_test, n_test]

    splits = data.random_split(raw_dataset, lengths,
                               generator=torch.Generator().manual_seed(42))

    torch.save(splits, "sequences.pt")


def main():
    make_all_sequences(8, 25, n_vocab=64)


if __name__ == '__main__':
    main()

