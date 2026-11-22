"""Train the Pop2Piano model with a pretrained audio encoder.

Examples
--------
Train from scratch:
    python train.py data/ --config config.yaml

Fine-tune a checkpoint (frozen MERT, new trainer state):
    python train.py data/ --finetune path/to/epoch=22-val_loss=1.2075.ckpt

Fine-tune and also train the top 4 MERT blocks:
    python train.py data/ --finetune path/to.ckpt --unfreeze-encoder --encoder-top-layers 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup

from data_roots import metadata_files
from dataset import Pop2PianoDataset, collate_batch, split_by_song
from transformer_wrapper import TransformerWrapper


# --------------------------------------------------------------------------- #
# Lightning module
# --------------------------------------------------------------------------- #
class TrainableTransformer(TransformerWrapper):
    def _step(self, batch):
        return self(batch).loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch["labels"].shape[0]
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch["labels"].shape[0],
        )
        return loss

    # ---- optimisation ---------------------------------------------------- #
    def _param_groups(self, weight_decay: float):
        """Parameter groups: (T5 + projections) and (fine-tuned MERT blocks).

        Biases, norm weights and other 1-D parameters (e.g. ``layer_logits``)
        get no weight decay. The MERT group only exists if some of MERT is
        trainable, and uses ``lr * encoder.lr_mult``.
        """
        mert_ids = {id(p) for p in self.audio_encoder.model.parameters()}
        buckets = {
            ("main", True): [],
            ("main", False): [],
            ("mert", True): [],
            ("mert", False): [],
        }
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            owner = "mert" if id(parameter) in mert_ids else "main"
            buckets[(owner, parameter.ndim >= 2)].append(parameter)

        mert_lr = self.lr * float(self.encoder_cfg.lr_mult)
        groups = []
        for (owner, decayed), params in buckets.items():
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": self.lr if owner == "main" else mert_lr,
                        "weight_decay": weight_decay if decayed else 0.0,
                    }
                )
        return groups

    def configure_optimizers(self):
        training = self.config.training
        name = str(training.optimizer).lower()

        if name == "adafactor":
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                self._param_groups(weight_decay=0.0),
                lr=self.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            optimizer = torch.optim.AdamW(
                self._param_groups(float(training.get("weight_decay", 0.01))),
                lr=self.lr,
            )

        if training.get("scheduler", "cosine") == "none":
            return optimizer

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training.get("warmup_steps", 500)),
            num_training_steps=int(self.trainer.estimated_stepping_batches),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pick(cli_value, config_value):
    """CLI value if given, else the config value."""
    return cli_value if cli_value is not None else config_value


def finetune_paths(checkpoint_path: Path, output_dir: str):
    """Checkpoint directory and CSV logger for a fine-tune.

    Fine-tunes are stored beside their source run:
        <.../version_N>/<checkpoint name>/finetuned
    or, if the checkpoint is not inside a ``version_N`` folder:
        <output_dir>/lightning_logs/<checkpoint name>/finetuned
    """
    checkpoint = checkpoint_path.resolve()
    root = Path(output_dir).resolve() / "lightning_logs"
    for parent in checkpoint.parents:
        if parent.name.startswith("version_") and parent.name[8:].isdigit():
            root = parent
            break

    checkpoint_dir = root / checkpoint.stem / "finetuned"
    logger = CSVLogger(save_dir=str(root), name=checkpoint.stem, version="finetuned")
    return checkpoint_dir, logger


def configure_encoder(model: TrainableTransformer, freeze: bool, top_layers: int):
    """Freeze MERT, or unfreeze only its top ``top_layers`` blocks."""
    encoder = model.audio_encoder
    if freeze:
        for parameter in encoder.model.parameters():
            parameter.requires_grad_(False)
        encoder.freeze = True
        encoder.model.eval()
        return "frozen"

    if top_layers < 1:
        raise ValueError("encoder_top_layers must be at least 1")
    n_tensors = encoder.unfreeze_top_layers(top_layers)
    return f"top {top_layers} MERT blocks trainable ({n_tensors} tensors)"


def build_datasets(args, config, seed):
    training = config.training
    composer = args.composer or str(training.get("composer", "composer1"))
    composer_value = config.composer_to_feature_token[composer]

    train_files, validation_files = split_by_song(
        metadata_files(args.data_dir),
        pick(args.validation_fraction, float(training.get("validation_fraction", 0.1))),
        seed,
    )
    train_dataset = Pop2PianoDataset(
        train_files,
        config,
        composer_value=composer_value,
        stride_fraction=pick(
            args.train_stride_fraction,
            float(training.get("train_stride_fraction", 0.5)),
        ),
        augment_pitch_shift=True,
    )
    validation_dataset = Pop2PianoDataset(
        validation_files,
        config,
        composer_value=composer_value,
        stride_fraction=1.0,
    )
    print(
        f"train: {len(train_files)} files / {len(train_dataset)} segments | "
        f"validation: {len(validation_files)} files / {len(validation_dataset)} segments"
    )
    return train_dataset, validation_dataset


def build_loaders(args, config, train_dataset, validation_dataset, seed):
    num_workers = pick(args.num_workers, int(config.training.get("num_workers", 4)))
    common = dict(
        batch_size=config.training.batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    sampler = None
    if args.data_weight:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"train: WeightedRandomSampler enabled ({len(train_dataset)} samples)")

    train_loader = DataLoader(
        train_dataset, sampler=sampler, shuffle=sampler is None, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def build_model(args, config, output_dir):
    """Returns (model, finetune_checkpoint_dir, finetune_logger)."""
    if not args.finetune:
        return TrainableTransformer(config), None, None

    training = config.training
    path = Path(args.finetune)
    if not path.is_file():
        raise FileNotFoundError(f"fine-tuning checkpoint not found: {path}")

    checkpoint_dir, logger = finetune_paths(path, output_dir)
    model = TrainableTransformer.load_from_checkpoint(str(path), config=config)
    model.lr = float(
        pick(args.finetune_lr, training.get("finetune_lr", training.lr))
    )

    if args.freeze_encoder:
        freeze = True
    elif args.unfreeze_encoder:
        freeze = False
    else:
        freeze = bool(training.get("freeze_encoder", True))
    top_layers = int(
        pick(args.encoder_top_layers, training.get("finetune_encoder_top_layers", 2))
    )
    encoder_state = configure_encoder(model, freeze, top_layers)

    print(
        f"Fine-tuning from {path}\n"
        f"  lr={model.lr:g}"
        + ("" if freeze else f" (MERT lr={model.lr * float(model.encoder_cfg.lr_mult):g})")
        + f" | audio encoder: {encoder_state}\n"
        f"  checkpoints -> {checkpoint_dir}"
    )
    return model, checkpoint_dir, logger


def build_trainer(args, config, output_dir, checkpoint_dir, logger):
    training = config.training
    use_gpu = torch.cuda.is_available()
    num_gpu = training.num_gpu if use_gpu else 1
    if use_gpu:
        torch.set_float32_matmul_precision("high")

    patience = pick(args.early_stopping_patience, training.get("early_stopping_patience", 10))
    min_delta = pick(args.early_stopping_min_delta, training.get("early_stopping_min_delta", 0.0))

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir) if checkpoint_dir is not None else None,
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename="{epoch}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if patience >= 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                min_delta=min_delta,
                verbose=True,
            )
        )

    kwargs = dict(
        accelerator="gpu" if use_gpu else "cpu",
        devices=num_gpu,
        max_epochs=args.max_epochs or training.max_epochs,
        accumulate_grad_batches=training.accumulate_grad_batches,
        gradient_clip_val=training.gradient_clip_val,
        check_val_every_n_epoch=training.check_val_every_n_epoch,
        log_every_n_steps=training.get("log_every_n_steps", 50),
        callbacks=callbacks,
    )
    if logger is not None:
        kwargs["logger"] = logger
    else:
        kwargs["default_root_dir"] = output_dir

    configured = training.get("precision")
    precision = pick(
        args.precision,
        str(configured) if configured not in (None, False) else None,
    )
    if not use_gpu and precision in ("bf16-mixed", "bf16"):
        precision = None
    if precision is not None:
        kwargs["precision"] = precision

    if use_gpu and num_gpu > 1:
        kwargs["strategy"] = "ddp"

    return pl.Trainer(**kwargs)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    config = OmegaConf.load(args.config)
    training = config.training

    resume = args.resume
    if resume is None and isinstance(training.get("resume", False), str):
        resume = str(training.resume)
    if args.finetune and resume is not None:
        raise ValueError("Use --finetune or --resume, not both")

    seed = pick(args.seed, int(training.get("seed", 3407)))
    output_dir = args.output_dir or str(training.get("output_dir", "checkpoints"))
    pl.seed_everything(seed, workers=True)

    train_dataset, validation_dataset = build_datasets(args, config, seed)
    train_loader, validation_loader = build_loaders(
        args, config, train_dataset, validation_dataset, seed
    )
    model, checkpoint_dir, logger = build_model(args, config, output_dir)
    trainer = build_trainer(args, config, output_dir, checkpoint_dir, logger)

    trainer.fit(model, train_loader, validation_loader, ckpt_path=resume)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pop2Piano.")
    parser.add_argument("data_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--composer", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="validation checks without val_loss improvement; -1 disables early stopping",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=None,
        help="minimum val_loss improvement required to reset patience",
    )
    parser.add_argument("--validation_fraction", type=float, default=None)
    parser.add_argument(
        "--train_stride_fraction",
        type=float,
        default=None,
        help="segment stride for training as a fraction of the segment length",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help='override config.training.precision (e.g. "bf16-mixed")',
    )
    parser.add_argument(
        "--data-weight",
        dest="data_weight",
        action="store_true",
        help="sample training segments proportionally to cleaner region scores",
    )
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")

    finetune = parser.add_argument_group("fine-tuning")
    finetune.add_argument(
        "--finetune",
        default=None,
        help="checkpoint whose weights to fine-tune; starts a new trainer state",
    )
    finetune.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="learning rate (default: training.finetune_lr or training.lr)",
    )
    encoder_group = finetune.add_mutually_exclusive_group()
    encoder_group.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="keep the pretrained audio encoder frozen",
    )
    encoder_group.add_argument(
        "--unfreeze-encoder",
        "--train-encoder",
        dest="unfreeze_encoder",
        action="store_true",
        help="unfreeze only the top MERT encoder blocks",
    )
    finetune.add_argument(
        "--encoder-top-layers",
        type=int,
        default=None,
        help="number of top MERT blocks to unfreeze (default: 2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())