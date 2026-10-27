import argparse
import glob
import os
import random

import librosa
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from midiaudiopair import MidiAudioPair
from midi_tokenizer import MidiTokenizer
from transformer_wrapper import TransformerWrapper
from data_roots import metadata_files


class Pop2PianoDataset(Dataset):
    def __init__(
        self,
        meta_files,
        config,
        composer_value,
        n_bars=None,
        max_target_length=None,
    ):
        self.config = config
        self.n_bars = config.dataset.n_bars if n_bars is None else n_bars
        self.steps_per_segment = self.n_bars * 4
        self.max_target_length = (
            config.dataset.target_length
            if max_target_length is None
            else max_target_length
        )
        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.composer_value = composer_value
        self.samples = []

        for meta_file in meta_files:
            sample = MidiAudioPair(meta_file)
            required = (
                getattr(sample, "song", None),
                getattr(sample, "beatstep", None),
                getattr(sample, "notes", None),
            )
            if all(path is not None and os.path.exists(path) for path in required):
                beatstep = np.load(sample.beatstep)
                for start in range(0, len(beatstep) - 1, self.steps_per_segment):
                    if start + self.steps_per_segment < len(beatstep):
                        self.samples.append((sample, start))

        if not self.samples:
            raise ValueError(
                "No valid training segments found. Run pop_align.py and "
                "bpm_quantize.py first."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, start = self.samples[index]
        beatstep = np.load(sample.beatstep)
        notes = np.load(sample.notes)

        end = start + self.steps_per_segment
        start_time = beatstep[start]
        end_time = beatstep[end]
        audio, _ = librosa.load(
            sample.song,
            sr=self.config.dataset.sample_rate,
            mono=True,
            offset=float(start_time),
            duration=float(end_time - start_time),
        )

        segment_notes = notes[
            (notes[:, 0] < end) & (notes[:, 1] > start)
        ].copy()
        segment_notes[:, 0] = np.maximum(segment_notes[:, 0], start) - start
        segment_notes[:, 1] = np.minimum(segment_notes[:, 1], end) - start
        tokens = self.tokenizer.notes_to_relative_tokens(
            segment_notes,
            offset_idx=0,
            add_eos=True,
        )
        tokens = tokens[: self.max_target_length]
        output = {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "labels": torch.tensor(tokens, dtype=torch.long),
            "composer": torch.tensor(self.composer_value, dtype=torch.long),
        }
        # print(output)
        return output


def collate_batch(items):
    audio = pad_sequence(
        [item["audio"] for item in items],
        batch_first=True,
        padding_value=0.0,
    )
    labels = pad_sequence(
        [item["labels"] for item in items],
        batch_first=True,
        padding_value=-100,
    )
    composer = torch.stack([item["composer"] for item in items])
    output = {"audio": audio, "labels": labels, "composer": composer}
    # print(output)
    return output


class TrainableTransformer(TransformerWrapper):
    def _forward_batch(self, batch):
        features = self.spectrogram(batch["audio"]).transpose(-1, -2)
        if self.mel_is_conditioned:
            features = self.mel_conditioner(features, batch["composer"])
        return self.transformer(inputs_embeds=features, labels=batch["labels"])

    def training_step(self, batch, batch_idx):
        loss = self._forward_batch(batch).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_batch(batch).loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.config.training.optimizer.lower() == "adafactor":
            from transformers.optimization import Adafactor

            optimizer = Adafactor(
                self.parameters(),
                lr=self.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def valid_meta_files(data_dir):
    return metadata_files(data_dir)


def main(args):
    pl.seed_everything(args.seed, workers=True)
    config = OmegaConf.load(args.config)
    composer_value = config.composer_to_feature_token[args.composer]
    files = valid_meta_files(args.data_dir)
    random.Random(args.seed).shuffle(files)

    split = max(1, int(len(files) * (1.0 - args.validation_fraction)))
    train_files = files[:split]
    validation_files = files[split:] or files[:1]

    train_dataset = Pop2PianoDataset(
        train_files,
        config,
        composer_value=composer_value,
    )
    validation_dataset = Pop2PianoDataset(
        validation_files,
        config,
        composer_value=composer_value,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    model = TrainableTransformer(config)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.training.num_gpu if torch.cuda.is_available() else 1,
        max_epochs=args.max_epochs or config.training.max_epochs,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        gradient_clip_val=config.training.gradient_clip_val,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        default_root_dir=args.output_dir,
    )
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pop2Piano.")
    parser.add_argument("data_dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--composer", default="composer1")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())
