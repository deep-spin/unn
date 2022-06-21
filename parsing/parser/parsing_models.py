import torch
from torch import nn

from transformers import BertModel
from biaffine import DeepBiaffineScorer, UNNBiaffineScorer
from nonproj import pack


class BertBiaffine(nn.Module):
    def __init__(self, hidden_dim, dropout_p, pretrained_weights, unn_iter=0, 
                 finetune_bert=False, bert_output_conf='last_hidden_state', bert_hidden_states=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_weights)
        self.bert_output_conf = bert_output_conf
        self.bert_hidden_states = bert_hidden_states

        if not finetune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        if unn_iter >= 0:  # switch to requiring <0, roadmap to deprecating it.
            self.scorer = UNNBiaffineScorer(
                input_size=self.bert.config.hidden_size,
                hidden_size=hidden_dim,
                dropout=dropout_p,
                n_iter=unn_iter)
        else:
            self.scorer = DeepBiaffineScorer(
                input_size=self.bert.config.hidden_size,
                hidden_size=hidden_dim,
                dropout=dropout_p)


    def forward(self, subwords, mapper, lengths, compute_energy=False):

        if self.bert_output_conf == 'last_hidden_state':
            # use last hidden state
            bert_output = self.bert(subwords, output_hidden_states=False)
            bert_subwords = bert_output["last_hidden_state"]
        else:  # 'hidden_states'
            # use average of last X states (outputed as tuple!)
            bert_output = self.bert(subwords, output_hidden_states=True)
            bert_subwords = sum(bert_output["hidden_states"][-self.bert_hidden_states:]) / self.bert_hidden_states

        bert_words = mapper.transpose(1, 2).to(dtype=bert_subwords.dtype) @ bert_subwords

        # treat CLS token (bert_words[0]) as contextual root token.
        emb_head = bert_words[:, :-1, :]
        emb_mod = bert_words[:, 1:-1, :]

        return self.scorer(emb_head, emb_mod, lengths, compute_energy)
