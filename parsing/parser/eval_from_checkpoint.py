import sys
import torch

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from ud_loaders import load_test_data
from nonproj import ParserAccuracy
from parsing_models import BertBiaffine
from conf import load_config


def validate(val_iter, device,  monitor_energy_every=0):
    acc = ParserAccuracy()
    model.eval()

    energies = []
    all_entropies = []

    for i, ex in enumerate(val_iter):
        words, mapper, _ = ex.word
        label, lengths = ex.head

        words, mapper, label, lengths = (x.to(device=device) for x in
                                         (words, mapper, label, lengths))

        compute_energy = (monitor_energy_every > 0
                          and i % monitor_energy_every == 0)
        res = model(words, mapper, lengths, compute_energy)

        if compute_energy:
            energies.append([i] + res.energies)
            print("val_energy", res.energies)
        all_entropies.extend(res.entropy.tolist())

        # compute accuracy terms and update overall sum
        acc.update(*res.accuracy_terms(label))

    return acc.value(), all_entropies


def eval(model, val_iter, test_iter):
    print('Validation:')
    (valid_acc, valid_allarc_acc, valid_fullsent_acc), valid_entr = validate(val_iter, device=conf['device'], monitor_energy_every=30)
    print('Accuracy:', valid_acc)
    print('All arcs accuracy:', valid_allarc_acc)
    print('Fullsent accuracy:', valid_fullsent_acc)

    print('Test:')
    (test_acc, test_allarc_acc, test_fullsent_acc), test_entr = validate(test_iter, device=conf['device'], monitor_energy_every=30)
    print('Accuracy:', test_acc)
    print('All arcs accuracy:', test_allarc_acc)
    print('Fullsent accuracy:', test_fullsent_acc)


if __name__ == '__main__':
    conf = load_config(show=True)
    
    tokenizer = BertTokenizer.from_pretrained(conf['bert'])

    model_path = f"saved_models/{conf['load_from_checkpoint']}"
    model = torch.load(model_path)
    model.scorer.n_iter=conf['unn_iter']
    
    if conf['device'] == 'cuda':
        model.cuda()

    val_iter, test_iter = load_test_data(conf['dataset'], tokenizer=tokenizer)

    print('Evaluating model', model_path)
    eval(model, val_iter, test_iter)




