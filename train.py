"""Train the Pop2Piano model with a pretrained audio encoder."""

import argparse

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from data_roots import metadata_files
from dataset import Pop2PianoDataset, collate_batch, split_by_song
from transformer_wrapper import TransformerWrapper


class TrainableTransformer(TransformerWrapper):
    def _step(self, batch):
        return self(batch).loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch["labels"].shape[0],
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

    def configure_optimizers(self):
        training = self.config.training
        encoder_params = [
            p for p in self.audio_encoder.model.parameters() if p.requires_grad
        ]
        encoder_ids = {id(p) for p in encoder_params}
        other_params = [
            p for p in self.parameters() if p.requires_grad and id(p) not in encoder_ids
        ]
        groups = [{"params": other_params, "lr": self.lr}]
        if encoder_params:  # only when the pretrained encoder is fine-tuned
            groups.append(
                {
                    "params": encoder_params,
                    "lr": self.lr * float(self.encoder_cfg.lr_mult),
                }
            )

        if training.optimizer.lower() == "adafactor":
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                groups,
                lr=self.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            optimizer = torch.optim.AdamW(
                groups, lr=self.lr, weight_decay=training.get("weight_decay", 0.01)
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


def main(args):
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    composer_value = config.composer_to_feature_token[args.composer]

    files = metadata_files(args.data_dir)
    train_files, validation_files = split_by_song(
        files, args.validation_fraction, args.seed
    )

    train_dataset = Pop2PianoDataset(
        train_files,
        config,
        composer_value=composer_value,
        stride_fraction=args.train_stride_fraction,
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

    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    model = TrainableTransformer(config)

    use_gpu = torch.cuda.is_available()
    num_gpu = config.training.num_gpu if use_gpu else 1
    if use_gpu:
        torch.set_float32_matmul_precision("high")

    trainer_kwargs = dict(
        accelerator="gpu" if use_gpu else "cpu",
        devices=num_gpu,
        max_epochs=args.max_epochs or config.training.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        log_every_n_steps=config.training.get("log_every_n_steps", 50),
        default_root_dir=args.output_dir,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
                filename="{epoch}-{val_loss:.4f}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    if args.precision is not None:
        trainer_kwargs["precision"] = args.precision
    if use_gpu and num_gpu > 1:
        trainer_kwargs["strategy"] = "ddp"

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_loader, validation_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pop2Piano.")
    parser.add_argument("data_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--composer", default="composer1")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument(
        "--train_stride_fraction",
        type=float,
        default=0.5,
        help="segment stride for training as a fraction of the segment length",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help='e.g. "bf16-mixed" (Lightning 2.x); default: Lightning default',
    )
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())