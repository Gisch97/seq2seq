import json 
import os
import time
import random
import numpy as np
import torch as tr
from datetime import datetime
import pandas as pd
import shutil
import pickle
from functools import partial
import mlflow
import mlflow.pytorch

from torch.utils.data import DataLoader
from .dataset import SeqDataset, pad_batch
from .model.unet import seq2seq 
from .embeddings import NT_DICT
from .parser import parser, get_parser_defaults
from .utils import  validate_file, read_train_file, read_test_file, read_pred_file 

def main():
    parser_defaults = get_parser_defaults()
    args = parser().parse_args()

    # Definir ruta de caché para entrenamiento
    cache_path = "cache/" if (not args.no_cache and args.command == "train") else None

    # Configuración global por defecto
    global_config = {
        "device": args.device,
        "batch_size": args.batch_size,
        "valid_split": 0.1,
        "max_len": 128,
        "verbose": not args.quiet,
        "cache_path": cache_path,
    }

    # Actualizar global_config a partir del archivo de configuración
    config_path = args.global_config or "config/global.json"
    try:
        with open(config_path) as f:
            global_config.update(json.load(f))
    except FileNotFoundError:
        if args.global_config:
            raise ValueError(f"Global configuration file not found: {args.global_config}")

    # Combinar defaults, configuración global y argumentos CLI (prioridad CLI)
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    final_config = {**parser_defaults, **global_config} #, **cli_args

    # Preparar directorio de caché si es necesario
    if final_config.get("cache_path"):
        shutil.rmtree(final_config["cache_path"], ignore_errors=True)
        os.makedirs(final_config["cache_path"], exist_ok=True)

     
    # Reproducibility
    tr.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
     
    mlflow.set_tracking_uri("sqlite:///results/mlflow/mlruns.db")
    artifact_location = "results/mlflow/mlruns/artifacts"

    try:
        mlflow.create_experiment(final_config["exp"], artifact_location=artifact_location )
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(final_config["exp"])
    with mlflow.start_run(run_name=final_config["run"]):
        if args.command == "train":
            read_train_file(args)  
            mlflow.log_params(final_config)                      
            mlflow.log_param("train_file",args.train_file)
            mlflow.log_param("valid_file",args.valid_file)
            mlflow.log_param("out_path",args.out_path)  
            mlflow.log_param("train_swaps",args.train_swaps)
            train(args.train_file, final_config, args.out_path,  args.valid_file, args.nworkers, args.train_swaps)
            

        if args.command == "test":
            read_test_file(args)
            test(args.test_file, args.model_weights, args.out_path, args.test_swaps ,final_config, args.nworkers) 
            mlflow.log_params(final_config) 
            mlflow.log_param("test_file",args.test_file) 
            mlflow.log_param("out_path",args.out_path) 
            mlflow.log_param("test_swaps",args.test_swaps)

        if args.command == "pred":
            read_pred_file(args)
            pred(args.pred_file, model_weights=args.model_weights, out_path=args.out_path, logits=args.logits, config=final_config, nworkers=args.nworkers, draw=args.draw, draw_resolution=args.draw_resolution)    
 
def train(train_file, config={}, out_path=None, valid_file=None, nworkers=2, train_swaps=0, verbose=True):
    # -----------------------------------------
    #   (1) Carga de datos y configuraciones
    # -----------------------------------------
    if out_path is None:
        out_path = f"results_{str(datetime.today()).replace(' ', '-')}/"
    else:
        out_path = out_path
    if verbose:
        print("Working on", out_path)

    if "cache_path" not in config:
        config["cache_path"] = "cache/"
    
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        raise ValueError(f"Output path {out_path} already exists")

    if valid_file is not None:
        train_file = train_file
        valid_file = valid_file
    else:
        data = pd.read_csv(train_file)
        valid_split = config["valid_split"] if "valid_split" in config else 0.1
        train_file = os.path.join(out_path, "train.csv")
        valid_file = os.path.join(out_path, "valid.csv")

        val_data = data.sample(frac = valid_split)
        val_data.to_csv(valid_file, index=False)
        data.drop(val_data.index).to_csv(train_file, index=False)
    
    pad_batch_with_fixed_length = partial(pad_batch, fixed_length=128)
    batch_size = config["batch_size"] if "batch_size" in config else 4
    valid_loader = DataLoader(
        SeqDataset(valid_file, **config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch_with_fixed_length,
    )
    
    if train_swaps > 0:
        config["train_noise"] = True
        config["test_noise"] = False
        config["swaps"] = train_swaps
    train_loader = DataLoader(
        SeqDataset(train_file, training=True, **config),
        batch_size=batch_size, 
        shuffle=True,
        num_workers=nworkers,
        collate_fn=pad_batch_with_fixed_length
    )    
    # -----------------------------------------
    #   (2) Inicio de entrenamiento
    # -----------------------------------------
    
    net = seq2seq(train_len=len(train_loader), **config)  
    mlflow.log_param("arc_num_params", sum(p.numel() for p in net.parameters()))  
      
    best_loss, patience_counter = np.inf, 0 
    patience = config["patience"] if "patience" in config else 30
    max_epochs = config["max_epochs"] if "max_epochs" in config else 1000
    logfile = os.path.join(out_path, "train_log.csv") 

    if verbose:
        print("Start training...")
    
    
    for epoch in range(max_epochs): 
        # --- Época: inicio ---
        epoch_start = time.time()
        
        # --- Fase de entrenamiento ---
        train_start = time.time()
        train_metrics = net.fit(train_loader)
        train_duration = time.time() - train_start
        mlflow.log_metric("train_time", train_duration, step=epoch)
        
        #  --- Registrar métricas de entrenamiento --- 
        for k, v in train_metrics.items(): 
            mlflow.log_metric(key=f"train_{k}", value=v, step=epoch)
        
        # --- Fase de validación ---
        val_metrics = net.test(valid_loader)
        for k, v in val_metrics.items():  
            mlflow.log_metric(key=f"valid_{k}", value=v, step=epoch)
        
          # --- Época: fin ---
        epoch_duration = time.time() - epoch_start
        mlflow.log_metric("epoch_time", epoch_duration, step=epoch)

        # ------------------------------
        # (3) Guardado de mejor modelo y paciencia
        # ------------------------------
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            tr.save(net.state_dict(), os.path.join(out_path, "weights.pmt"))
            patience_counter = 0
            mlflow.log_metric(key=f"best_epoch", value=epoch)
            
        else:
            mlflow.log_metric(key=f"valid_{k}", value=v, step=epoch)
            patience_counter += 1
            if patience_counter > patience:
                break
        # ------------------------------
        # (4) Append CSV de log
        # ------------------------------
        if not os.path.exists(logfile):
            with open(logfile, "w") as f: 
                msg = ','.join(['epoch']+[f"train_{k}" for k in sorted(train_metrics.keys())]+[f"valid_{k}" for k in sorted(val_metrics.keys())]) + "\n"
                f.write(msg)
                f.flush()
                if verbose:
                    print(msg)

        with open(logfile, "a") as f: 
            msg = ','.join([str(epoch)]+[f'{train_metrics[k]:.4f}' for k in sorted(train_metrics.keys())]+[f'{val_metrics[k]:.4f}' for k in sorted(val_metrics.keys())]) + "\n"
            f.write(msg)
            f.flush()    
            if verbose:
                print(msg)
            
    # ------------------------------
    # (5) Limpieza y loggeado final
    # ------------------------------         
    shutil.rmtree(config["cache_path"], ignore_errors=True)
    
    tmp_file = os.path.join(out_path, "train.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    tmp_file = os.path.join(out_path, "valid.csv")
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
     
    mlflow.pytorch.log_model(net, "model")
        
 
    
def test(test_file, model_weights=None, output_file=None, test_swaps=0, config={}, nworkers=2, verbose=True):
    test_file = test_file
    test_file = validate_file(test_file)
    if verbose not in config:
        config["verbose"] = verbose
    if test_swaps > 0:
        config["test_noise"] = True
        config["swaps"] = test_swaps
        
    pad_batch_with_fixed_length = partial(pad_batch, fixed_length=128)
    test_loader = DataLoader(
        SeqDataset(test_file, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=nworkers,   
        collate_fn=pad_batch_with_fixed_length,
    )
    if model_weights is not None:
        net = seq2seq(weights=model_weights, **config)
    else:
        net = seq2seq(pretrained=True, **config)
    
    mlflow.log_param("arc_num_params", sum(p.numel() for p in net.parameters()))
    if verbose:
        print(f"Start test of {test_file}")        
    test_metrics = net.test(test_loader)

    for k, v in test_metrics.items():  
        mlflow.log_metric(key=f"test_{k}", value=v)
    summary = ",".join([k for k in sorted(test_metrics.keys())]) + "\n" + ",".join([f"{test_metrics[k]:.3f}" for k in sorted(test_metrics.keys())])+ "\n" 
    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(summary)
    if verbose:
        print(summary)

def pred(pred_input, sequence_id='pred_id', model_weights=None, out_path=None, logits=False, config={}, nworkers=2,  verbose=True):
    
    if out_path is None:
        output_format = "text"
    else:
        _, ext = os.path.splitext(out_path)
        if ext == "":
            if os.path.isdir(out_path):
                raise ValueError(f"Output path {out_path} already exists")
            os.makedirs(out_path)
            output_format = "ct"
        elif ext != ".csv":
            raise ValueError(f"Output path must be a .csv file or a folder, not {ext}")
        else:
            output_format = "csv"

    file_input = os.path.isfile(pred_input)
    if file_input:
        pred_file = validate_file(pred_input)
    else:
        pred_input = pred_input.upper().strip()
        nt_set = set([i for item  in list(NT_DICT.values()) for i in item] + list(NT_DICT.keys()))
        if set(pred_input).issubset(nt_set):
            pred_file = f"{sequence_id}.csv"
            with open(pred_file, "w") as f:
                f.write("id,sequence\n")
                f.write(f"{sequence_id},{pred_input}\n")
            
        else:
            raise ValueError(f"Invalid input nt {set(pred_input)}, either the file is missing or the secuence have invalid nucleotides (should be any of {nt_set})")
        
    pad_batch_with_fixed_length = partial(pad_batch, fixed_length=128)
    pred_loader = DataLoader(
        SeqDataset(pred_file, for_prediction=True, **config),
        batch_size=config["batch_size"] if "batch_size" in config else 4,
        shuffle=False,
        num_workers=nworkers,
        collate_fn=pad_batch_with_fixed_length,
    )
    
    if model_weights is not None:
        weights = model_weights
        net = seq2seq(weights=weights, **config)
    else:
        net = seq2seq(pretrained=True, **config)

    if verbose:        
        print(f"Start prediction of {pred_file}")

    predictions, logits_list = net.pred(pred_loader, logits=logits)
   
    if not file_input:
        os.remove(pred_file)

    if output_format == "text":
        for i in range(len(predictions)):
            item = predictions.iloc[i]
            print(item.id)
            print(item.sequence)
    elif output_format == "csv":
        predictions.to_csv(out_path, index=False)
    else: # ct
        for i in range(len(predictions)):
            item = predictions.iloc[i]
        base = os.path.split(out_path)[0] if not os.path.isdir(out_path) else out_path
        if len(base) == 0:
            base = "."
        out_path_dir = base + "/logits/"
        os.mkdir(out_path_dir)
        for id, pred, pred_post in logits_list:
            pickle.dump((pred, pred_post), open(os.path.join(out_path_dir, id + ".pk"), "wb"))
