# imports
import os
import subprocess as sp
from platform import system
import warnings
import numpy as np
import torch as tr
import pandas as pd
import json
from .embeddings import NT_DICT, VOCABULARY


# All possible matching brackets for base pairing
MATCHING_BRACKETS = [
    ["(", ")"],
    ["[", "]"],
    ["{", "}"],
    ["<", ">"],
    ["A", "a"],
    ["B", "a"],
]
# Normalization.
BRACKET_DICT = {"!": "A", "?": "a", "C": "B", "D": "b"}



def valid_sequence(seq):
    """Check if sequence is valid"""
    return set(seq.upper()) <= (set(NT_DICT.keys()).union(set(VOCABULARY)))


def validate_file(pred_file):
    """Validate input file fasta/csv format and return csv file"""
    if os.path.splitext(pred_file)[1] == ".fasta":
        table = []
        with open(pred_file) as f:
            row = []  # id, seq, (optionally) struct
            for line in f:
                if line.startswith(">"):
                    if row:
                        table.append(row)
                        row = []
                    row.append(line[1:].strip())
                else:
                    if len(row) == 1:  # then is seq
                        row.append(line.strip())
                        if not valid_sequence(row[-1]):
                            raise ValueError(
                                f"Sequence {row.upper()} contains invalid characters"
                            )
                    else:  # struct
                        row.append(
                            line.strip()[: len(row[1])]
                        )  # some fasta formats have extra information in the structure line
        if row:
            table.append(row)

        pred_file = pred_file.replace(".fasta", ".csv")

        if len(table[-1]) == 2:
            columns = ["id", "sequence"]
        else:
            columns = ["id", "sequence", "dotbracket"]

        pd.DataFrame(table, columns=columns).to_csv(pred_file, index=False)

    elif os.path.splitext(pred_file)[1] != ".csv":
        raise ValueError(
            "Predicting from a file with format different from .csv or .fasta is not supported"
        )

    return pred_file


def apply_config(args, config_attr, default_path, error_msg):
    config_val = getattr(args, config_attr)
    # Se utiliza el valor especificado solo si no es None; de lo contrario se usa la ruta por defecto
    config_path = config_val if config_val is not None else default_path
    try:
        with open(config_path) as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key):
                current_val = getattr(args, key)
                # Se actualiza solo si el valor es None o cadena vacía
                if current_val is None or current_val == "":
                    setattr(args, key, value)
    except FileNotFoundError:
        raise ValueError(error_msg)


def read_train_file(args):
    if args.train_file is None:
        apply_config(
            args, "train_config", "config/train.json", "No train_file specified"
        )


def read_test_file(args):
    if args.test_file is None:
        apply_config(args, "test_config", "config/test.json", "No test_file specified")


def read_pred_file(args):
    if args.pred_file is None:
        apply_config(args, "pred_config", "config/pred.json", "No pred_file specified")


def merge_configs(global_config, parsed_args):
    """
    Fusiona el archivo de configuración con los argumentos parseados.
    La prioridad es:
    1. Argumentos parseados (CI / línea de comandos).
    2. Archivo de configuración.
    3. Valores por defecto.
    """
    final_config = {}
    for key, value in parser_defaults.items():
        final_config[key] = value  # Inicializa con los valores por defecto del parser

    for key, value in global_config.items():
        if value is not None:
            final_config[key] = (
                value  # Actualiza con valores del archivo de configuración
            )

    for arg_key, arg_value in vars(parsed_args).items():
        if arg_value is not None:  # Si el argumento fue proporcionado en la CI
            final_config[arg_key] = arg_value

    return final_config
