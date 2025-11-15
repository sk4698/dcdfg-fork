# This code is heavily adopted from the run_gaussian.py script of the original
# DCD-FG repository (https://github.com/Genentech/dcdfg) to use with custom data.

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split

from dcdfg.callback import (
    AugLagrangianCallback,
    ConditionalEarlyStopping,
    CustomProgressBar,
)
from dcdfg.dcdi.model import MLPGaussianModel
from dcdfg.linear_baseline.model import LinearGaussianModel
from dcdfg.lowrank_linear_baseline.model import LinearModuleGaussianModel
from dcdfg.lowrank_mlp.model import MLPModuleGaussianModel
from dcdfg.simulation_data import SimulationDataset


def train_dcdfg(
    train_dataset: SimulationDataset,
    train_size: float = 0.8,
    model_type: str = "mlplr",
    train_batch_size: int = 64,
    lr: float = 1e-3,
    reg_coeff: float = 0.1,
    constraint_mode: str = "exp",
    num_modules: int = 20,
    poly: bool = False,
    num_gpus: int = 1,
    num_train_epochs: int = 60000,
    fine_tune: bool = False,
    num_fine_epochs: int = 200,
) -> pl.LightningModule:
    """
    Train the DCD-FG model on the given dataset and given parameters.
    This function mirrors the logic in `run_gaussian.py` but is callable
    directly and does not use any Weights & Biases logging.

    Args:
        train_dataset: Dataset to train on (will be split into train/validation).
        train_size: Fraction of the dataset to use for training (rest for validation).
        model_type: Type of model to train. One of "linear", "linearlr", "mlplr", "dcdi".
        train_batch_size: Batch size for training.
        lr: Learning rate.
        reg_coeff: Regularization coefficient.
        constraint_mode: Constraint mode.
        num_modules: Number of modules (used for low-rank models).
        poly: Whether to use a polynomial on the linear model (only for "linear").
        num_gpus: Number of GPUs to use.
        num_train_epochs: Number of training epochs.
        fine_tune: Whether to fine-tune the model (second training stage).
        num_fine_epochs: Number of fine-tuning epochs.

    Returns:
        model: Trained model.
    """
    # Determine number of nodes from the dataset
    nb_nodes = train_dataset.dim

    # Split into train/validation subsets
    n_total = len(train_dataset)
    n_train = int(train_size * n_total)
    n_val = n_total - n_train
    train_subset, val_subset = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_subset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_subset, batch_size=256, num_workers=4)

    # Instantiate model (mirrors logic from run_gaussian.py)
    if model_type == "linear":
        model = LinearGaussianModel(
            nb_nodes,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
            poly=poly,
        )
    elif model_type == "linearlr":
        model = LinearModuleGaussianModel(
            nb_nodes,
            num_modules,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    elif model_type == "mlplr":
        model = MLPModuleGaussianModel(
            nb_nodes,
            2,
            num_modules,
            16,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    elif model_type == "dcdi":
        model = MLPGaussianModel(
            nb_nodes,
            2,
            16,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    # -----------------------------
    # Step 1: augmented Lagrangian
    # -----------------------------
    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=num_gpus,
        max_epochs=num_train_epochs,
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
    )
    trainer.fit(model, train_loader, val_loader)

    # -----------------------------
    # Step 2: fine-tune (optional)
    # -----------------------------
    if fine_tune:
        # Freeze and prune adjacency
        model.module.threshold()

        # Ensure constraint mode is compatible during fine-tuning
        # (mirrors the comment in run_gaussian.py)
        model.module.constraint_mode = "exp"
        # remove DAG constraints: prediction-only problem
        model.gamma = 0.0
        model.mu = 0.0

        early_stop_2_callback = EarlyStopping(
            monitor="Val/nll",
            min_delta=1e-6,
            patience=5,
            verbose=True,
            mode="min",
        )
        trainer_fine = pl.Trainer(
            gpus=num_gpus,
            max_epochs=num_fine_epochs,
            val_check_interval=1.0,
            callbacks=[early_stop_2_callback, CustomProgressBar()],
        )
        trainer_fine.fit(model, train_loader, val_loader)

    return model

def load_model_from_weights(
    model_weights_path: str,
    num_nodes: int,
    model_type: str,
    num_modules: int = 20,
    poly: bool = False,
    lr: float = 1e-3,
    reg_coeff: float = 0.1,
    constraint_mode: str = "exp",
) -> pl.LightningModule:
    """
    Load a model from weights and return it.
    Args:
        model_weights_path: Path to the model weights.
        num_nodes: Number of nodes in the graph.
        model_type: Type of model to load.
        num_modules: Number of modules (used for low-rank models).
        poly: Whether to use a polynomial on the linear model (only for "linear").
        lr: Learning rate.
        reg_coeff: Regularization coefficient.
        constraint_mode: Constraint mode.
    """
    # Instantiate model (mirrors logic from run_gaussian.py)
    if model_type == "linear":
        model = LinearGaussianModel(
            num_nodes,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
            poly=poly,
        )
    elif model_type == "linearlr":
        model = LinearModuleGaussianModel(
            num_nodes,
            num_modules,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    elif model_type == "mlplr":
        model = MLPModuleGaussianModel(
            num_nodes,
            2,
            num_modules,
            16,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    elif model_type == "dcdi":
        model = MLPGaussianModel(
            num_nodes,
            2,
            16,
            lr_init=lr,
            reg_coeff=reg_coeff,
            constraint_mode=constraint_mode,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.load_state_dict(torch.load(model_weights_path))
    return model