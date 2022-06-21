# Undirected Neural Networks

This repository contains the code for the paper "Modeling Structure with Undirected Neural Networks" by Tsvetomila Mihaylova, Vlad Niculae and André Martins. accepted at ICML 2022.

## Contents

* Folder *attn* contains experiments for sequence completion with undirected self-attention.

* Folder *mnist* contains experiments for image classification and visualization with undirected convolutional networks.

* Folder *parsing* contains the experiments for parsing with structured UNN.


## Requirements

* Python 3 (>3.6; prefered 3.7)

* Install the latest PyTorch version

* Execute `pip install -r requirements.txt` to install the required libraries.


There are additional requirements for the parsing experiments, please check the file `parsing/parser/README.md`.


## Citation

@misc{unn2022,
  url = {https://arxiv.org/abs/2202.03760},
  author = {Mihaylova, Tsvetomila and Niculae, Vlad and Martins, Andr\'{e} F.T.},
  title = {Modeling Structure with Undirected Neural Networks},
  year = {2022},
}


