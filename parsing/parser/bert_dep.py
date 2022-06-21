#!/usr/bin/env python3

import sys, os
import random
import torch

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

# relative imports
from ud_loaders import load_data
from nonproj import ParserAccuracy
from parsing_models import BertBiaffine
from conf import load_config

# torch.autograd.set_detect_anomaly(True)

# validation and train loops
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

    if energies:
        cols = ["i"] + [str(k) for k in range(len(energies[0]) - 1)]
        art = wandb.Artifact("energies", type="energies")
        energy_table = wandb.Table(columns=cols, data=energies)
        art.add(energy_table, "energies")
        wandb.log_artifact(art)

    model.train()
    return acc.value(), all_entropies


def train(train_iter, val_iter, model, conf):
    finetune_bert = conf['finetune_bert']
    n_steps = conf['epochs'] * len(train_iter)
    opt = AdamW(model.parameters(), lr=conf['lr'], eps=1e-8)
    if finetune_bert:
        scheduler = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=conf['num_warmup_steps'],
            num_training_steps=n_steps)

    max_acc = -1
    max_acc_epoch = -1
    max_acc_last_model_path = None
    for epoch in range(conf['epochs']):

        model.train()
        total_loss = 0
        for i, ex in enumerate(train_iter):
            opt.zero_grad()
            words, mapper, _ = ex.word
            label, lengths = ex.head

            words, mapper, label, lengths = (
                x.to(device=conf['device'])
                for x in (words, mapper, label, lengths)
            )

            res = model(words, mapper, lengths)
            log_prob = res.log_prob(label)
            loss = -log_prob.mean()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           conf['grad_clip'])
            opt.step()
            if finetune_bert:
                scheduler.step()

        (train_acc, train_allarc_acc, train_fullsent_acc), train_entr = validate(train_iter, device=conf['device'])
        (valid_acc, valid_allarc_acc, valid_fullsent_acc), valid_entr = validate(val_iter, device=conf['device'],
                                         monitor_energy_every=30)

        print(
            f"Epoch {epoch} "
            f"train {train_acc:.4f} "
            f"valid {valid_acc:.4f} "
            f"train_allarc {train_allarc_acc:.4f} "
            f"valid_allarc {valid_allarc_acc:.4f} "
            f"train_fullsent {train_fullsent_acc:.4f} "
            f"valid_fullcent {valid_fullsent_acc:.4f} "
            f"loss {total_loss:.2f}"
        )

        # discard nan entropies, since this is just for histogram purposes
        train_entr = [ent for ent in train_entr if ent == ent]
        valid_entr = [ent for ent in valid_entr if ent == ent]

        log_result = wandb.log({
            'train_acc': train_acc,
            'loss': total_loss,
            'valid_acc': valid_acc,
            'train_allarc_acc': train_allarc_acc,
            'valid_allarc_acc': valid_allarc_acc,
            'train_fullsent_acc': train_fullsent_acc,
            'valid_fullsent_acc': valid_fullsent_acc,
            'max_valid_acc': max_acc,
            'max_valid_epoch': max_acc_epoch,
            'train_entropies': wandb.Histogram(train_entr),
            'valid_entropies': wandb.Histogram(valid_entr),
        })

        model_path = f"saved_models/{conf['project']}-{wandb.run.name}.pt"
        print('model_path', model_path)

        if max_acc < valid_acc:
            max_acc = valid_acc
            max_acc_epoch = epoch
            previous_best_path = max_acc_last_model_path
            max_acc_last_model_path = model_path

            if previous_best_path:
                os.remove(previous_best_path)

            torch.save(model, model_path)
            


    print(f'RESULT1: Best DEV Accuracy: {max_acc}. Epoch: {max_acc_epoch}')


if __name__ == '__main__':

    conf = load_config(show=True)
    run = wandb.init(project=conf['project'], entity=conf['entity'], config=conf)

    torch.manual_seed(conf['seed'])
    random.seed(conf['seed'])

    tokenizer = BertTokenizer.from_pretrained(conf['bert'])

    model = BertBiaffine(hidden_dim=conf['hidden'],
                         dropout_p=conf['p_drop'],
                         pretrained_weights=conf['bert'],
                         unn_iter=conf['unn_iter'],
                         finetune_bert=conf['finetune_bert'],
                         bert_output_conf=conf['bert_output'],
                         bert_hidden_states=conf['bert_hidden_states'])

    if conf['device'] == 'cuda':
        model.cuda()

    wandb.watch(model.scorer, log_freq=1)

    train_iter, val_iter = load_data(
        conf['dataset'],
        batch_size=conf['batch_size'],
        tokenizer=tokenizer)

    train(train_iter, val_iter, model, conf)

    print("=== Finished. ===")
