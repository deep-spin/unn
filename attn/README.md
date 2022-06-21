# Undirected Self-Attention

## Data Generation

In order to generate numerical sequences, run the file `generate_sequences.pt`. This will save a file `sequences.pt` which can be passed as an input for training the models.

## Running

In order to execute the undirected self-attention experiments, run the file `attn_run.py` with the following parameters:

* `dataset` - the saved dataset file, for example `sequences.pt`.

* `k` - the number of iterations; for random order, set zero or negative value. if the value is negative, the inference iterations are equal to `-k`.

* `order` - this is the number of operations. Valid values are `default` and `random`.


Run with:

`python attn_run.py sequences.pt 1 default`


## Attention Plots

Run the file `read_from_checkpoint.py` and specify the model path in the variable `model_path` in the code.

Executing the file will plot the attention weights for the first element from the test set.
