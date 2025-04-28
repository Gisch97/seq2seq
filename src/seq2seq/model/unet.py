import pandas as pd
import math
from torch import nn
from torch.nn.functional import mse_loss, cross_entropy
import torch as tr
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from .conv_layers import N_Conv, UpBlock, DownBlock, OutConv
from ..metrics import compute_metrics
from .._version import __version__


def seq2motif(weights=None, **kwargs):
    """
    seq2motif: a deep learning-based autoencoder for RNA sequence connections predictions.
    weights (str): Path to weights file
    **kwargs: Model hyperparameters
    """

    model = Seq2Motif(**kwargs)
    if weights is not None:
        print(f"Load weights from {weights}")
        model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
    else:
        print("No weights provided, using random initialization")
    model.log_model()
    mlflow.set_tag("model", "seq2motif-pseudo_probing")
    return model


class Seq2Motif(nn.Module):
    def __init__(
        self,
        train_len=0,
        device="cpu",
        lr=1e-3,
        scheduler="none", 
        verbose=True,
        train_noise=False,
        test_noise=False,
        **kwargs,
    ):
        """Base instantiation of model"""
        super().__init__()

        self.device = device
        self.verbose = verbose
        self.config = kwargs
        self.test_noise = test_noise
        self.train_noise = train_noise
        self.hyperparameters = {
            "hyp_device": device,
            "hyp_lr": lr,
            "hyp_scheduler": scheduler,
            "hyp_verbose": verbose,
        }
        # Define architecture
        self.build_graph(**kwargs)
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # lr scheduler
        self.scheduler_name = scheduler
        if scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        elif scheduler == "cycle":
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                steps_per_epoch=train_len,
                epochs=self.config["max_epochs"],
            )
        else:
            self.scheduler = None

        self.to(device)

    def build_graph(
        self,
        embedding_dim=4,
        num_conv=2,
        pool_mode="max",
        up_mode="upsample",
        skip=1,
        addition="cat",
        features=[4, 8, 16, 32, 64],
        **kwargs,
    ):
        
        features = [4]
        n_4=2
        n_8=1
        for _ in range(n_4):
            features.append(4)
        for _ in range(n_8):
            features.append(8)

        rev_features = features[::-1]
        encoder_blocks = len(features) - 1
        self.L_min = 128 // ((2**encoder_blocks))
        volume = [(128 / 2**i) * f for i, f in enumerate(features)]
        self.out_dim = 1
        self.architecture = {
            "arc_embedding_dim": embedding_dim,
            "arc_encoder_blocks": encoder_blocks,
            "arc_initial_volume": 4 * 128,
            "arc_latent_volume": volume[-1],
            "arc_features": features,
            "arc_num_conv": num_conv,
            "arc_pool_mode": pool_mode,
            "arc_up_mode": up_mode,
            "arc_addition": addition,
            "arc_skip": skip,
        }
        self.inc = N_Conv(embedding_dim, features[0], num_conv)

        self.down = nn.ModuleList(
            [
                DownBlock(
                    in_channels=features[i],
                    out_channels=features[i + 1],
                    num_conv=num_conv,
                    pool_mode=pool_mode,
                )
                for i in range(encoder_blocks)
            ]
        )
        self.up = nn.ModuleList(
            [
                UpBlock(
                    in_channels=rev_features[i],
                    out_channels=rev_features[i + 1],
                    num_conv=num_conv,
                    up_mode=up_mode,
                    addition=addition,
                    skip=skip,
                )
                for i in range(len(rev_features) - 1)
            ]
        )
        self.outc = OutConv(features[0], self.out_dim)

    def forward(self, x):
        x = x.to(self.device)
        x = self.inc(x)
        encoder_outputs = [x]
        for i, down in enumerate(self.down):
            x = down(x)
            encoder_outputs.append(x)

        x_latent = x

        skips = encoder_outputs[:-1][::-1]
        for up, skip in zip(self.up, skips):
            x = up(x, skip)

        y_pred = tr.sigmoid(self.outc(x))

        return y_pred, x_latent

    def loss_func(self, y_pred, y):
        """yhat and y are [N, L]"""
        y = y.view(y.shape[0], -1)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        return mse_loss(y_pred, y)

    def fit(self, loader):
        self.train()
        
        metrics = {"loss": 0, "F1": 0, "Accuracy": 0, "Accuracy_seq": 0}
        if self.verbose:
            loader = tqdm(loader)
 
        for batch in loader:
            y = batch["pseudo_probing"].to(self.device)
            x_model = batch['embedding'].to(self.device)         
            mask = batch["mask"].to(self.device) 
            
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            y_pred, _ = self(x_model)
            print(y_pred)
            print(y)
            loss = self.loss_func(y_pred, y)
            metrics["loss"] += loss.item()

            batch_metrics = compute_metrics(y_pred, y, mask, binary=True)
            for k, v in batch_metrics.items():
                metrics[k] += v

            loss.backward()
            self.optimizer.step()

            if self.scheduler_name == "cycle":
                self.scheduler.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()

        metrics = {"loss": 0, "F1": 0, "Accuracy": 0, "Accuracy_seq": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:
                y = batch["pseudo_probing"].to(self.device)
                x_model = batch["embedding"].to(self.device)
                mask = batch["mask"].to(self.device) 

                y_pred, z = self(x_model)
                loss = self.loss_func(y_pred, y)
                metrics["loss"] += loss.item()
                batch_metrics = compute_metrics(y_pred, y, mask, binary=True)
                
                
                
                for k, v in batch_metrics.items():
                    metrics[k] += v

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def pred(self, loader, logits=False):
        self.eval()

        if self.verbose:
            loader = tqdm(loader)

        predictions, logits_list = [], []
        with tr.no_grad():
            for batch in loader:

                seqid = batch["id"]
                embedding = batch["embedding"]
                sequences = batch["sequence"]
                lengths = batch["length"]
                pseudo_probing = batch["pseudo_probing"],
                motif_emb = batch["motif_emb"],      
                y_pred, z = self(embedding)

                for k in range(y_pred.shape[0]):
                    seq_len = lengths[k]

                    predictions.append(
                        (
                            seqid[k],
                            sequences[k],
                            seq_len,
                            embedding[k, :, :seq_len].cpu().numpy(),
                            pseudo_probing[k, :, :seq_len].cpu().numpy(),
                            motif_emb[k, :, :seq_len].cpu().numpy(),
                            y_pred[k, :, :seq_len].cpu().numpy(),
                            z[k].cpu().numpy(),
                        )
                    )

        predictions = pd.DataFrame(
            predictions,
            columns=[
                "id",
                "sequence",
                "length",
                "embedding",
                "pseudo_probing",
                "motif_emb",
                "connection_pred",
                "latent",
            ],
        )

        return predictions, logits_list

    def log_model(self):
        """Logs the model architecture and hyperparameters to MLflow."""
        mlflow.log_params(self.hyperparameters)
        mlflow.log_params(self.architecture)
