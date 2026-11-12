# Configuration reference

This document describes every parameter currently defined in
[`config.yaml`](./config.yaml). The training script loads this YAML file with
OmegaConf; there is no `config.py` file in this project.

## Project

| Parameter | Value | Description |
|---|---:|---|
| `project` | `pop2piano` | Project identifier used as configuration metadata. |

## Dataset

| Parameter | Value | Description |
|---|---:|---|
| `dataset.target_length` | `256` | Maximum number of MIDI tokens generated for one training segment. It is also used as the default maximum generation length during inference. Increase it only when the model and memory budget support longer sequences. |
| `dataset.n_bars` | `4` | Number of musical bars in each training segment. The tokenizer uses four beat steps per bar, so the default segment contains 16 beat steps. |
| `dataset.sample_rate` | `22050` | Sample rate of the rendered mix used by inference and audio utilities. Training audio is resampled for the configured encoder separately. |
| `dataset.mel_is_conditioned` | `true` | Enables composer conditioning through the mel/audio conditioning path. Set to `false` to disable this conditioning. |

## Composer feature tokens

`composer_to_feature_token` maps each composer name to the special token
inserted for composer conditioning. The `--composer` argument passed to
`train.py` must be one of these keys.

| Composer | Token |
|---|---:|
| `composer1` | `2052` |
| `composer2` | `2053` |
| `composer3` | `2054` |
| `composer4` | `2055` |
| `composer5` | `2056` |
| `composer6` | `2057` |
| `composer7` | `2058` |
| `composer8` | `2059` |
| `composer9` | `2060` |
| `composer10` | `2061` |
| `composer11` | `2062` |
| `composer12` | `2063` |
| `composer13` | `2064` |
| `composer14` | `2065` |
| `composer15` | `2066` |
| `composer16` | `2067` |
| `composer17` | `2068` |
| `composer18` | `2069` |
| `composer19` | `2070` |
| `composer20` | `2071` |
| `composer21` | `2072` |

## T5 model

These values override the corresponding Hugging Face T5 configuration.

| Parameter | Value | Description |
|---|---:|---|
| `t5.feed_forward_proj` | `gated-gelu` | Feed-forward projection used by each T5 block. |
| `t5.tie_word_embeddings` | `false` | Keeps the input and output token embeddings as separate parameters. |
| `t5.tie_encoder_decoder` | `false` | Keeps encoder and decoder embeddings separate. |
| `t5.vocab_size` | `288` | Total token vocabulary available to the model. It must be at least as large as the tokenizer vocabulary. |
| `t5.relative_attention_num_buckets` | `32` | Number of buckets used by T5 relative-position attention. More buckets provide finer distance distinctions at additional model cost. |

## Tokenizer vocabulary

The tokenizer divides the vocabulary into token categories:

| Parameter | Value | Description |
|---|---:|---|
| `tokenizer.vocab_size.special` | `4` | Reserved special tokens. |
| `tokenizer.vocab_size.note` | `128` | One token for each MIDI pitch from 0 through 127. |
| `tokenizer.vocab_size.time` | `100` | Time-position tokens. This must cover the number of beat steps in a segment. |
| `tokenizer.vocab_size.duration` | `32` | Duration tokens for note lengths. |
| `tokenizer.vocab_size.velocity_bins` | `0` | Number of velocity/dynamics bins. `0` disables velocity tokens. |

The configured T5 vocabulary size must be large enough for the complete
tokenizer vocabulary and any special conditioning tokens.

## Training

| Parameter | Value | Description |
|---|---:|---|
| `training.resume` | `false` | Legacy configuration flag for resuming. The current CLI uses `--resume` to select a checkpoint path. |
| `training.num_gpu` | `1` | Number of GPUs used when CUDA is available. Multiple GPUs use distributed data parallel training. |
| `training.max_epochs` | `10000` | Maximum number of training epochs. Early stopping can end training sooner. |
| `training.early_stopping_patience` | `10` | Number of validation checks without improvement in `val_loss` before stopping. |
| `training.early_stopping_min_delta` | `0.0` | Minimum decrease in `val_loss` required to count as an improvement. |
| `training.accumulate_grad_batches` | `2` | Number of batches whose gradients are accumulated before an optimizer update. |
| `training.check_val_every_n_epoch` | `1` | Run validation after every epoch. This also determines how often early stopping is evaluated. |
| `training.find_lr` | `false` | Legacy flag; the current `train.py` does not run an automatic learning-rate finder. |
| `training.optimizer` | `adafactor` | Optimizer selection. `adafactor` uses Hugging Face Adafactor; any other value selects AdamW. |
| `training.lr` | `0.0001` | Base learning rate for trainable non-encoder parameters. |
| `training.lr_min` | `0.000001` | Configured minimum learning rate. The current training scheduler does not directly use this value. |
| `training.batch_size` | `16` | Number of examples per DataLoader batch on each GPU. |
| `training.gradient_clip_val` | `3.0` | Maximum gradient norm used for gradient clipping. |
| `training.warmup_steps` | `500` | Number of optimizer steps used for linear learning-rate warmup. |
| `training.weight_decay` | `0.01` | AdamW weight decay. Adafactor does not use this value in the current implementation. |
| `training.scheduler` | `cosine` | Learning-rate scheduler. `cosine` enables warmup followed by cosine decay; `none` keeps the learning rate constant. |
| `training.log_every_n_steps` | `10` | Training-log interval in optimizer/training steps. |

The effective batch size is:

```text
training.batch_size
× training.accumulate_grad_batches
× training.num_gpu
```

With the current values and one GPU, it is `16 × 2 × 1 = 32`.

## Audio encoder

| Parameter | Value | Description |
|---|---:|---|
| `encoder.name` | `m-a-p/MERT-v1-95M` | Hugging Face pretrained audio encoder used to condition the T5 model. |
| `encoder.revision` | `null` | Optional Hugging Face model revision or commit. `null` uses the repository default revision. |
| `encoder.sample_rate` | `24000` | Audio sample rate expected by MERT. Keep this aligned with the encoder’s requirements. |
| `encoder.freeze` | `true` | Freezes MERT weights. Set to `false` to fine-tune the encoder. |
| `encoder.normalize_waveform` | `true` | Normalizes waveform input before passing it to the encoder. |
| `encoder.context_seconds` | `2.0` | Audio context included on each side of a target segment. |
| `encoder.lr_mult` | `0.1` | Learning-rate multiplier for MERT parameters when `encoder.freeze` is `false`. The encoder learning rate becomes `training.lr × encoder.lr_mult`. |
