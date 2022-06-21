import sys, csv
import torch

from omegaconf import OmegaConf
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

from ud_loaders import load_test_data
from nonproj import ParserAccuracy
from parsing_models import BertBiaffine


def validate(model, val_iter, device,  monitor_energy_every=0):
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


def eval(model, conf, val_iter, test_iter):
    result = {}
    (valid_acc, valid_allarc_acc, valid_fullsent_acc), valid_entr = validate(model, val_iter, device=conf['device'], monitor_energy_every=30)
    result['valid_acc'] = valid_acc
    result['valid_allarc_acc'] = valid_allarc_acc.item()
    result['valid_fullsent_acc'] = valid_fullsent_acc

    (test_acc, test_allarc_acc, test_fullsent_acc), test_entr = validate(model, test_iter, device=conf['device'], monitor_energy_every=30)
    result['test_acc'] = test_acc
    result['test_allarc_acc'] = test_allarc_acc.item()
    result['test_fullsent_acc'] = test_fullsent_acc

    print(result)
    return result


def eval_on_checkpoint(lang, unn_iter, wandb_run):
    # load config for given language and checkpoint run name
    baseconf=f"configs/{lang}.yaml"
    conf = OmegaConf.load(baseconf)
    checkpoint = f"{lang}-{wandb_run}.pt"
    # # Load the tokenizer    
    tokenizer = BertTokenizer.from_pretrained(conf['bert'])

    # Load the model from saved checkpoint
    model_path = f"saved_models/{checkpoint}"
    model = torch.load(model_path)

    model.scorer.n_iter = unn_iter

    if conf['device'] == 'cuda':
        model.cuda()

    val_iter, test_iter = load_test_data(conf['dataset'], tokenizer=tokenizer)

    print('Evaluating model', model_path)
    result = eval(model, conf, val_iter, test_iter)

    return result


def write_line_to_file(filename, *text):
    text = ','.join(map(str, text))
    print(text)
    with open(filename, 'a', encoding="utf-8") as out:
        out.write(text)
        out.write('\n')


if __name__ == '__main__':
    _, file_in, file_out = sys.argv

    write_line_to_file(file_out, 'lang', 'unn_iter', 'wandb_name',
                    'test_acc', 'test_allarc_acc', 'test_fullsent_acc',
                    'valid_acc', 'valid_allarc_acc', 'valid_fullsent_acc')

    with open(file_in, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            lang = row[0]
            unn_iter = int(row[1])
            wandb_run = row[2]

            result = eval_on_checkpoint(lang, unn_iter, wandb_run)

            write_line_to_file(file_out, lang, str(unn_iter), wandb_run,
                            result['test_acc'], result['test_allarc_acc'], result['test_fullsent_acc'],
                            result['valid_acc'], result['valid_allarc_acc'], result['valid_fullsent_acc'])


