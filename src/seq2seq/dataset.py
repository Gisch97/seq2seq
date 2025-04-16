import pandas as pd
from torch.utils.data import Dataset
import torch as tr
import os
import json
import pickle
import random
from .embeddings import OneHotEmbedding
from typing import Union



class SeqDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        min_len=0,
        max_len=512,
        verbose=False,
        cache_path=None,
        for_prediction=False,
        training=False,
        swaps=0,
        mode='perc',
        **kargs,
    ):
        """
        interaction_prior: none, probmat
        """
        self.max_len = max_len
        self.verbose = verbose
        if cache_path is not None and not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache = cache_path

        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.training = training

        assert (
            "sequence" in data.columns and "id" in data.columns
        ), "Dataset should contain 'id' and 'sequence' columns"

        data["len"] = data.sequence.str.len()

        if max_len is None:
            max_len = max(data.len)
        self.max_len = max_len

        datalen = len(data)

        data = data[(data.len >= min_len) & (data.len <= max_len)]

        if len(data) < datalen:
            print(
                f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
            )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        self.embedding = OneHotEmbedding()
        self.embedding_size = self.embedding.emb_size
        self.swaps = swaps
        self.mode = mode
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seqid = self.ids[idx]
        cache = f"{self.cache}/{seqid}.pk"
        if (self.cache is not None) and os.path.isfile(cache):
            item = pickle.load(open(cache, "rb"))
        else:
            sequence = self.sequences[idx]
            L = len(sequence)
            seq_emb = self.embedding.seq2emb(sequence)
            if self.swaps == 0:
                embedding_with_noise = seq_emb
            elif self.swaps != 0:
                embedding_with_noise = add_noise(seq_emb.clone(), self.swaps)
            else:
                print('ERROR: swaps < 0')

            item = {
                "id": seqid,
                "length": L,
                "sequence": sequence,
                "embedding": seq_emb,
                "embedding_with_noise": embedding_with_noise,
            }

            if self.cache is not None:
                pickle.dump(item, open(cache, "wb"))

        return item


def pad_batch(batch, fixed_length=0):
    """batch is a dictionary with different variables lists"""
    L = [b["length"] for b in batch]
    if fixed_length == 0:
        fixed_length = max(L)
    embedding_pad = tr.zeros((len(batch), batch[0]["embedding"].shape[0], fixed_length))
    embedding_pad_w_noise = tr.zeros(
        (len(batch), batch[0]["embedding_with_noise"].shape[0], fixed_length)
    )
    mask = tr.zeros((len(batch), fixed_length), dtype=tr.bool)

    for k in range(len(batch)):
        embedding_pad[k, :, : L[k]] = batch[k]["embedding"]
        embedding_pad_w_noise[k, :, : L[k]] = batch[k]["embedding_with_noise"]
        mask[k, : L[k]] = 1

    out_batch = {
        "id": [b["id"] for b in batch],
        "length": L,
        "sequence": [b["sequence"] for b in batch],
        "embedding": embedding_pad,
        "embedding_with_noise": embedding_pad_w_noise,
        "mask": mask,
    }

    return out_batch


def add_noise(x: tr.Tensor, swaps: Union[int,float] = 0, mode: str = 'perc') -> tr.Tensor:
    """
    Introduces noise into the tensor `x` by modifying some of its columns with random values.

    Parameters:
    x (torch.Tensor): Input tensor of size (N, len), where N=4 is the number of One-Hot dimension and len is the number of nucleotides.
    swaps (int or float, optional): Number of swaps to perform. Can be:
        - A positive integer, specifying the exact number of swaps.
        - A value between 0 and 1, interpreted as a percentage of the number of columns.
        - 0 (default), meaning no changes are made.
    mode (str, optional): Interpretation mode of `swaps`.
        - 'perc' (default): If `swaps` is 1, all columns are replaced.
        - 'int': `swaps` is treated as an absolute number of swaps.

    Returns:
    torch.Tensor: Modified tensor with added noise.
    """
    len = x.shape[-1]
    if swaps == 0:
        return x
    elif swaps > 0 and swaps < 1: 
        swaps = round(len * swaps)
    elif (mode == 'perc' and swaps == 1) or swaps > len:
        swaps = len
    
    rand_nt = [_ for _ in range(len)]
    random.shuffle(rand_nt)
    v = [0, 1, 2, 3] 
    
    for i in range(swaps): 
        nt = tr.zeros([4], dtype=tr.float)
        random.shuffle(v)
        nt[v[0]] = 1.0
        x[:, rand_nt[i]] = nt
    return x
