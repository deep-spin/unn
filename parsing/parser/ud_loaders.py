import torch
import torchtext.legacy.data as data


INDEX_COL_WORD = 1
INDEX_COL_HEAD = 6


def len_filt_train(x): return len(x.word[0]) == len(x.word[1]) and len(x.word[2]) > 4 and len(x.word[0]) < 100

def len_filt_valid(x): return len(x.word[0]) == len(x.word[1]) and len(x.word[2]) > 0


def token_pre(tokenizer, q):
    st = " ".join(q)
    s = tokenizer.tokenize(st)

    out = [0]
    cur = 0
    expect = ""
    first = True

    for i, w in enumerate(s):
        if len(expect) == 0:
            cur += 1
            
            if cur <= len(q):
                expect = q[cur - 1].lower()
            else:
                expect = q[-1]

            first = True
        if w.startswith("##"):
            out.append(-1)
            expect = expect[len(w) - 2 :]
        elif first:
            out.append(cur)
            expect = '' if w == '[UNK]' else expect[len(w) :]
            first = False
        else:
            expect = expect[len(w) :]

    out.append(cur + 1)
    # assert cur == len(q)-1, "%s %s \n%s\n%s"%(len(q), cur, q, s)

    token_ixs = tokenizer.encode(st, add_special_tokens=True)

    if cur != len(q):
        print("error", cur, len(q))
        # return [0] * (len(q) + 2), [0] * (len(q) + 2), q
        return token_ixs, [0] * (len(out)), q
    # else:
    #     if len(q) < 10:
    #         print('-------------')
    #         print(q)
    #         print(len(token_ixs), len(out))
    #         print(out)

    
    return token_ixs, out, q


def token_post(ls):
    lengths = [len(l[0]) for l in ls]

    length = max(lengths)
    out = [l[0] + ([0] * (length - len(l[0]))) for l in ls]

    lengths2 = [max(l[1]) + 1 for l in ls]
    length2 = max(lengths2)
    out2 = torch.zeros(len(ls), length, length2)
    for b, l in enumerate(ls):
        for i, w in enumerate(l[1]):
            if w != -1:
                out2[b, i, w] = 1
    return torch.LongTensor(out), out2.long(), lengths


def SubTokenizedField(tokenizer):
    """
    Field for use with pytorch-transformer
    """
    FIELD = data.RawField(
        preprocessing=lambda s: token_pre(tokenizer, s), postprocessing=token_post
    )
    FIELD.is_target = False
    return FIELD


class ConllXDataset(data.Dataset):
    def __init__(self, path, fields, encoding="utf-8", separator="\t", **kwargs):
        examples = []
        columns = [[], []]
        column_map = {INDEX_COL_WORD: 0, INDEX_COL_HEAD: 1}
        with open(path, encoding=encoding) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = [[], []]
                else:

                    conll_cols = line.split(separator)

                    # skip lines where the first field is not an int
                    skip = False
                    try:
                        int(conll_cols[0])
                    except ValueError:
                        skip = True

                    if skip:
                        continue

                    for col, i in column_map.items():
                        columns[i].append(conll_cols[col])

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(ConllXDataset, self).__init__(examples, fields, **kwargs)


def batch_num(nums):
    lengths = torch.tensor([len(n) for n in nums]).long()
    n = lengths.max()
    out = torch.zeros(len(nums), n).long()
    for b, n in enumerate(nums):
        out[b, :len(n)] = torch.tensor(n)
    return out, lengths


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def load_data(data_prefix, batch_size, tokenizer):

    # define fields
    head_field = data.RawField(
        preprocessing=lambda x: [safe_cast(i, int, 0) for i in x],
        postprocessing=batch_num
    )
    head_field.is_target = True
    word_field = SubTokenizedField(tokenizer)
    fields = (('word', word_field), ('head', head_field))

    # define datasets and iterators over them
    train = ConllXDataset(f'{data_prefix}-train.conllu',
                          fields,
                          filter_pred=len_filt_train)

    train_iter = data.BucketIterator(train, batch_size=batch_size)

    val = ConllXDataset(f'{data_prefix}-dev.conllu',
                        fields,
                        filter_pred=len_filt_valid)
    val_iter = data.BucketIterator(val, batch_size=1)

    return train_iter, val_iter


def load_test_data(data_prefix, tokenizer):
    # define fields
    head_field = data.RawField(
        preprocessing=lambda x: [safe_cast(i, int, 0) for i in x],
        postprocessing=batch_num
    )
    head_field.is_target = True
    word_field = SubTokenizedField(tokenizer)
    fields = (('word', word_field), ('head', head_field))

    # define datasets and iterators over them
    val = ConllXDataset(f'{data_prefix}-dev.conllu',
                        fields,
                        filter_pred=len_filt_valid)
    val_iter = data.BucketIterator(val, batch_size=1)

    test = ConllXDataset(f'{data_prefix}-test.conllu',
                        fields,
                        filter_pred=len_filt_valid)
    test_iter = data.BucketIterator(test, batch_size=1)

    return val_iter, test_iter

    
