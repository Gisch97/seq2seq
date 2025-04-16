
# **AEseq2seq - dev**

This is the development repository for AEseq2seq, a RNA autoencoder converting sequences into sequences, originally forked from the [sincFold](https://github.com/sinc-lab/sincFold) . For model tracking, MLflow is used, with a sqlite backend.
```bash
 mlflow ui --backend-store-uri sqlite:///mlruns.db --host 0.0.0.0 --port 5000
```


## Install

You can clone the repository with:

    git clone https://github.com/Gisch97/AEseq2seq.git
    cd sincFold/

and install with:

    pip install .

on Windows, you will probably need to add the python scripts folder to the PATH. It should work with python 3.9-3.11.

## Predicting sequences

To predict the secondary structure of a sequence using the pretrained weights:
    
    seq2seq pred AACCGGGUCAGGUCCGGAAGGAAGCAGCCCUAA
 
 

## Input files

The usage of the seq2seq model has the following input priority:

1. Command line interface
2. Input files
3. Default inputs

__vesion__ = 0.1