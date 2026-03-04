#
# Self-supervised training entry point for Rasouli PPG dataset
#

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from training.configs.training_config import TrainingPpgSsl
from training.datasets.minimal_dataset import RasouliPPGDataset
from training.models.efficientnet import EfficientNet
from training.models.ssl_wrapper import (
    MomentumSslMultiViewModelWrapper as SslMultiViewModelWrapper,
)
from training.objectives.info_nce_reg import InfoNceRegLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised training on Rasouli PPG dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to Rasouli PPG dataset file "
             "(np.load-able, list/array of dicts or DataFrame-like).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="ppg-ssl-rasouli",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional).",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logging.info(f"Logging to {log_file}")


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(state, checkpoint_path)
    logging.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
):
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    map_location = device if device is not None else "cpu"
    state = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    logging.info(f"Resumed from epoch {epoch}, global_step {global_step}")
    return epoch, global_step


def maybe_init_wandb(args, model, optimizer):
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError:
        logging.warning("wandb is not installed; disabling wandb logging.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "epochs": args.epochs,
            "batch_size": TrainingPpgSsl.dataloader_params.get("batch_size", 16),
            "optimizer": TrainingPpgSsl.optimizer_name,
            "optimizer_params": TrainingPpgSsl.optimizer_params,
            "scheduler": TrainingPpgSsl.scheduler_name,
            "scheduler_params": TrainingPpgSsl.scheduler_params,
            "model_name": TrainingPpgSsl.model_name,
            "loss_name": TrainingPpgSsl.loss_name,
        },
    )
    wandb.watch(model, log="all", log_freq=100)
    # Optional: log optimizer config explicitly
    return run


def create_scheduler(optimizer):
    name = getattr(TrainingPpgSsl, "scheduler_name", None)
    params = getattr(TrainingPpgSsl, "scheduler_params", {}) or {}

    if name == "step_lr":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(
            optimizer,
            step_size=params.get("step_size", 125),
            gamma=params.get("gamma", 0.5),
        )
    # Add more schedulers here if needed.
    return None


def set_seed():
    seed = getattr(TrainingPpgSsl, "seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    device = torch.device(args.device)

    root = Path(os.getcwd())
    log_dir = root / args.log_dir
    checkpoint_dir = root / args.checkpoint_dir

    setup_logging(log_dir)
    set_seed()

    deterministic = getattr(TrainingPpgSsl, "deterministic_execution", False)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        logging.info("Using deterministic execution.")

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    dataset_cfg = TrainingPpgSsl.dataset_params

    dataset = RasouliPPGDataset(
        data_path=args.data_path,
        do_zscore=dataset_cfg.get("do_zscore", True),
        augmentation_name=dataset_cfg.get("augmentation_name", "identity"),
        augmentation_config=dataset_cfg.get("augmentation_config", {}),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=TrainingPpgSsl.dataloader_params.get("batch_size", 16),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # -----------------------------
    # Model: backbone + SSL wrapper
    # -----------------------------
    backbone_cfg = TrainingPpgSsl.model_params["backbone_params"]
    backbone = EfficientNet(**backbone_cfg)

    wrapper_cfg = TrainingPpgSsl.model_params["wrapper_params"]
    model = SslMultiViewModelWrapper(backbone=backbone, **wrapper_cfg)
    model = model.to(device)

    # -----------------------------
    # Loss
    # -----------------------------
    loss_cfg = TrainingPpgSsl.loss_params
    criterion = InfoNceRegLoss(**loss_cfg).to(device)

    # -----------------------------
    # Optimizer & Scheduler
    # -----------------------------
    opt_cfg = TrainingPpgSsl.optimizer_params
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )
    scheduler = create_scheduler(optimizer)

    # -----------------------------
    # wandb
    # -----------------------------
    wandb_run = maybe_init_wandb(args, model, optimizer)
    if wandb_run is not None:
        import wandb  # safe: only if successfully imported earlier

    # -----------------------------
    # Resume (optional)
    # -----------------------------
    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, device=device
            )
            start_epoch += 1  # start from next epoch
        else:
            logging.warning(f"Checkpoint {ckpt_path} not found; starting from scratch.")

    # -----------------------------
    # Training loop
    # -----------------------------
    model.train()
    print_interval = max(1, getattr(TrainingPpgSsl, "log_interval", 1))
    checkpoint_interval = getattr(TrainingPpgSsl, "checkpoint_interval", 1)
    momentum_base = TrainingPpgSsl.train_type_params["momentum_params"][
        "momentum_value_base"
    ]

    for epoch in range(start_epoch, args.epochs + 1):
        for batch_idx, (views, pids, indices) in enumerate(dataloader):
            views = views.to(device, non_blocking=True)
            pids = torch.as_tensor(pids, device=device, dtype=torch.long)

            optimizer.zero_grad()

            # Forward pass through momentum SSL wrapper
            output_student, output_teacher = model(views)

            # Compute InfoNCE + regularization loss
            loss = criterion(output_student, output_teacher, pids=pids)

            # Backpropagation and optimizer step
            loss.backward()
            optimizer.step()

            # Momentum update for teacher network
            model.momentum_update(momentum_value=momentum_base)

            global_step += 1

            if batch_idx % print_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Step {global_step} "
                    f"LR {lr:.6f} "
                    f"Loss {loss.item():.4f}"
                )
                logging.info(msg)

                if args.use_wandb and wandb_run is not None:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )

        # Step scheduler per epoch, if any
        if scheduler is not None:
            scheduler.step()

        # Save checkpoint at configured interval
        if (epoch % checkpoint_interval) == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch}_step_{global_step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step)

    logging.info("Training finished.")

    if args.use_wandb and wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()