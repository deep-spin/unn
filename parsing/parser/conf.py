import torch
from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class ParserConfigSchema:
    baseconf: str
    project: str # the wandb project
    entity: str # the wandb entity
    seed: int
    lr: float
    bert: str
    hidden: int
    p_drop: float
    dataset: str
    grad_clip: float
    num_warmup_steps: int
    batch_size: int
    epochs: int
    unn_iter: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    finetune_bert: bool = False 
    bert_output: str = 'last_hidden_state' # possible values: 'last_hidden_state', 'hidden_states', 
    bert_hidden_states: int = 4 # only used if bert_output='hidden_states'
    load_from_checkpoint: str = '' # only for evaluation of saved models


def load_config(show=False):
    conf = OmegaConf.from_cli()

    # merge with base config yaml from disk.
    # cli flags take priority.
    if 'baseconf' in conf:
        baseconf = OmegaConf.load(conf.baseconf)
        conf = OmegaConf.merge(baseconf, conf)

    # validate against schema
    schema = OmegaConf.structured(ParserConfigSchema)
    conf = OmegaConf.merge(schema, conf)

    if show:
        print(OmegaConf.to_yaml(conf))

    conf = OmegaConf.to_container(conf)

    return conf
