# Training guide: pop audio → piano MIDI

This document explains how training works with `transformer_wrapper.py`, `dataset.py` and `train.py`: the model, the inputs, the dataset, the configuration and the training loop.

---

## 1. Overview

The task is a **sequence-to-sequence** problem: given a short, beat-aligned excerpt of a pop song, the model generates the tokens of a piano cover of that excerpt.

```
                 song audio excerpt (+ context)              beat grid        composer
                          │                                      │                │
                          ▼                                      │                │
              ┌───────────────────────┐                          │                │
              │ MERT (pretrained)     │                          │                │
              │ 13 layers, 75 Hz      │                          │                │
              └───────────┬───────────┘                          │                │
                 learned weighted sum of layers                  │                │
                          ▼                                      ▼                │
                LayerNorm → Linear → (+ beat embedding) → LayerNorm               │
                          │                                                       │
                          └────────────── prepend composer frame ◄────────────────┘
                                           │  (+ attention mask)
                                           ▼
                         T5 encoder  ──►  T5 decoder  ──►  onset/duration MIDI tokens
                       (from scratch)     (from scratch)    (cross-entropy loss)
```

- **Input**: MERT features of the excerpt, a beat-position embedding, and one composer frame.
- **Output**: a sequence of tokens `TIME PITCH DURATION PITCH DURATION … TIME …`, ending with EOS (see section 3.8). They are produced by `MidiTokenizer`. A segment owns the notes that **start** inside it.
- **Loss**: token-level cross-entropy with teacher forcing (the standard Hugging Face T5 loss).

| File | Role |
|---|---|
| `transformer_wrapper.py` | The model (`TransformerWrapper`) and the inference code (`generate`) |
| `dataset.py` | `Pop2PianoDataset`, `collate_batch`, and the song-level train/validation split |
| `train.py` | `TrainableTransformer` (loss, optimizer, scheduler) and the training script |

---

## 2. The data

### 2.1 What the dataset expects

The training data is the output of the cleaner (or any dataset with the same layout). `data_roots.metadata_files(data_dir)` lists the metadata files (`*.yaml`). For each one, `MidiAudioPair` exposes the paths used here:

| Attribute | File | Used for |
|---|---|---|
| `sample.song` | `<id>.pitchshift.wav` | Song audio (the model input) |
| `sample.beatstep` | `<id>.beatstep.npy` | Beat grid in seconds |
| `sample.notes` | `<id>.notes.npy` | Piano notes (`N × 4`; columns 0 and 1 are the start and end beat-step indices) |

Only the **song audio** is used as input. The MIDI is used through `notes.npy`, which becomes the target tokens.

### 2.2 Which samples are kept

A sample is skipped when:

- `sample.is_valid` is false,
- it is a cleaner chunk with `eval.kept` explicitly set to `False` (a missing flag means "kept", since the cleaner does not write it),
- the song, beatstep or notes file is missing,
- `notes.npy` is empty.

### 2.3 Segmentation

Each file is cut into **segments of `n_bars * 4` beat steps**:

```
beat steps:  0 1 2 3 4 5 6 7 8 9 ...
segment 0:   [─────────────]
segment 1:         [─────────────]      ← stride = segment length × stride_fraction
```

- A segment is `start … start + n_bars*4` on the beat grid, so its duration in seconds depends on the tempo. With `steps_per_beat = 2`, 120 BPM and `n_bars = 2`, a segment is 8 steps ≈ 2 s.
- Segments never cross the end of a file, so they never cross a cleaner-chunk boundary.
- Segments with fewer than `min_segment_notes` (default 1) notes are skipped.
- **Training** uses `--train_stride_fraction 0.5` (default): 50 % overlapping segments, so the same audio is seen with different beat phases. **Validation** uses 1.0 (disjoint segments).

At start-up the dataset loads every `beatstep.npy` and `notes.npy` once to build the segment list, which can take a while on large datasets.

### 2.3.1 Train/validation split

`split_by_song` groups files by **source song** (`eval.source_song_ytid` for cleaner chunks, otherwise `song.ytid`) and assigns whole songs to train or validation (`--validation_fraction`, default 0.1, shuffled with `--seed`). This matters because chunks of one song share audio and overlap in content; splitting by file would leak training data into validation and make `val_loss` look better than it is.

### 2.4 One dataset item

`Pop2PianoDataset.__getitem__` returns:

| Field | Shape / type | Meaning |
|---|---|---|
| `audio` | `(L,)` float32, encoder sample rate (24 kHz for MERT) | The segment **plus `context_seconds` on each side**, truncated at the file's start and end |
| `offset` | float, seconds | Where the segment starts inside `audio` |
| `duration` | float, seconds | Length of the segment |
| `beat_grid` | `(S+1,)` float32, `S = n_bars*4` | Beat-step times relative to the segment start (`beat_grid[0] = 0`) |
| `labels` | `(T,)` long | Target tokens from `tokenizer.segment_to_tokens(notes, start, end)`; segments longer than `target_length` are discarded at start-up |
| `composer` | long | Composer token value from `composer_to_feature_token` |

Only the notes whose **onset** lies in the segment are tokenized, with their real durations (capped at the duration vocabulary). Notes held over from the previous segment are not repeated, and nothing is clipped at the segment end.

### 2.5 Batching (`collate_batch`)

| Batch field | Description |
|---|---|
| `audio` | `(B, L_max)` zero-padded waveforms |
| `audio_lengths` | `(B,)` real lengths in samples |
| `offsets`, `durations` | `(B,)` seconds |
| `beat_grids` | `(B, S+1)` |
| `labels` | `(B, T_max)` padded with `-100` (ignored by the loss) |
| `composer` | `(B,)` |

---

## 3. The model

### 3.1 Audio encoder (pretrained, `PretrainedAudioEncoder`)

- **Model**: MERT (default `m-a-p/MERT-v1-95M`), loaded with `AutoModel.from_pretrained(..., trust_remote_code=True)`. It expects 24 kHz audio and outputs frames at **75 Hz** (one frame per 320 samples, 13.3 ms), 768 dimensions.
- **Pre-processing**: each waveform is normalized (zero mean, unit variance over its *real* length), then zero-padded. No attention mask is given to MERT, which is what this family of models expects, so the output for a window depends slightly on the other windows' lengths in the same batch.
- **Layer mixing**: the hidden states of all layers (embedding + 12 transformer layers) are combined with a learned softmax-weighted sum (`layer_logits`, initialised uniform). This lets the model find which layers carry the most useful information for transcription.
- **Frozen by default** (`encoder.freeze: true`): MERT weights get no gradients and the module stays in `eval()` mode even during training (no dropout). Only the layer-mixing weights and everything after MERT are trained.

### 3.2 Context window

Each segment is encoded together with `context_seconds` of audio on each side. After encoding, **only the frames whose centre time falls inside the segment** are kept:

```
window:   |──ctx──|═══ segment ═══|──ctx──|
frames:    ......   ● ● ● ● ● ●     ......     (only these go to T5)
```

This gives every frame knowledge of its neighbours, which reduces discontinuities between consecutive segments. In training the context is limited to what the file contains (a cleaner chunk is ~8 s long), so segments at the edge of a chunk see less context than in inference. At inference, the same procedure is applied to the full song.

### 3.3 Beat-position embedding

For every kept frame at time `τ` (seconds from the segment start), the model computes its position on the beat grid:

- `idx`: the beat step containing `τ`, and `phase`: how far through that step it is (0–1),
- `position = idx + phase` (a continuous position in steps),
- features: `sin/cos(2π · position / P)` for `P ∈ {1, 2, 4, 8, 16, 32}` steps, plus `log(step_duration)` as a tempo cue → 13 features,
- a `Linear(13 → d_model)` projects them, and the result is **added** to the projected audio features.

This tells the encoder where each frame sits on the same time axis the output tokens use. Because the features are periodic and continuous, any `n_bars` works at inference. The position is relative to the segment start: there is no downbeat information.

### 3.4 Composer frame

If composer conditioning is on (`dataset.mel_is_conditioned`, default true), a learned embedding of the composer (`composer_value − min(values)`) is **prepended as one extra frame** of the encoder sequence. If it is off, no composer frame is added.

### 3.5 Input assembly

```
frames  = LayerNorm(Linear(LayerNorm(mixed MERT features))) + Linear(beat features)
frames  = LayerNorm(frames)
sequence = [composer frame] + frames            (padded to the longest in the batch)
attention_mask = 1 for the composer frame and real frames, 0 for padding
```

The sequence is given to T5 as `inputs_embeds` together with `attention_mask`, so padding is never attended to.

### 3.6 T5 encoder-decoder

- A `T5ForConditionalGeneration` built from the `t5-small` config (downloaded from the hub) with every key of `config.t5` applied on top, then **initialised from scratch** (only MERT is pretrained).
- The encoder reads the sequence above. The decoder generates the tokens autoregressively.
- Vocabulary: `[special 4][pitch 128][time][duration][velocity bins]`. `config.t5.vocab_size` must be at least `tokenizer.vocab_total` (264 with the defaults); the model refuses to start otherwise.

### 3.7 What is trained

| Component | Trained | Notes |
|---|---|---|
| MERT | no (default) | Set `encoder.freeze: false` to fine-tune it with `lr × lr_mult` |
| `layer_logits` | yes | Layer-mixing weights |
| `feat_norm`, `audio_proj`, `beat_proj`, `input_norm` | yes | Adapters between MERT and T5 |
| `composer_embedding` | yes | |
| T5 encoder and decoder | yes | From scratch |

### 3.8 Tokenizer (`midi_tokenizer.py`)

A target is the sequence of the notes that start inside the segment, ordered by onset then pitch:

```
TIME(t)  PITCH(p) DURATION(d)  PITCH(p) DURATION(d)  ...  TIME(t')  PITCH(p) DURATION(d) ...  EOS
```

- `TIME(t)`: absolute beat-step position of the onset inside the segment, emitted once per distinct onset (notes of a chord share it).
- `DURATION(d)`: note length in beat steps (1 to `vocab_size.duration`, longer notes are capped). It is never clipped at the segment end: the audio context (`context_seconds`) lets the model see what follows.
- `VELOCITY(v)` (optional, `velocity_bins > 0`): emitted before a note only when the bin changes. Off by default; every note then gets velocity 77.
- It costs about 20-25 % fewer tokens than the previous on/off scheme on dense segments (2 tokens per note + 1 per distinct onset), and there are no note-off, velocity-toggle or offset-time tokens.
- Decoding never fails on damaged output: a pitch without a duration is ignored, repeated notes keep the longest duration, and onsets outside the chunk are dropped.

Because a chunk only writes the notes that start in it, neighbouring chunks never cut or re-strike a held note, which the previous tokenizer did at every chunk border.

---

## 4. The training loop (`train.py`)

1. **Seed** (`--seed`), load the config, resolve the composer value from `config.composer_to_feature_token[--composer]`.
2. **Split by song**, build the train and validation datasets and loaders (shuffle on for train, `persistent_workers` when `--num_workers > 0`).
3. **Build the model** (`TrainableTransformer`, a `TransformerWrapper` with the training methods).
4. **Each step**: `self(batch)` → encoder inputs and mask → `T5(inputs_embeds, attention_mask, labels)` → cross-entropy loss, logged as `train_loss`. Validation logs `val_loss` (with `sync_dist=True`).
5. **Optimizer**: `AdamW` (weight decay `training.weight_decay`, default 0.01), or `Adafactor` (no relative step, no scaling) if `training.optimizer` is `adafactor`. Parameter groups: all trainable weights at `lr`, plus MERT at `lr × encoder.lr_mult` if it is unfrozen.
6. **Scheduler**: linear warmup for `training.warmup_steps` (default 500) then cosine decay over `trainer.estimated_stepping_batches`, stepped every optimizer step. Set `training.scheduler: none` for a constant LR.
7. **Trainer**:
   - GPU if available, `training.num_gpu` devices, DDP when more than one,
   - `accumulate_grad_batches`, `gradient_clip_val`, `check_val_every_n_epoch`, `log_every_n_steps` from the config,
   - optional mixed precision with `--precision` (for example `bf16-mixed`),
   - `ModelCheckpoint` on `val_loss` (best 3 + last), early stopping on `val_loss`, and a learning-rate monitor,
   - resume with `--resume path/to.ckpt`.

The **effective batch size** is `training.batch_size × accumulate_grad_batches × num_gpu`.

Checkpoints include the frozen MERT weights, so they are larger than the trainable part alone.

---

## 5. Configuration reference

### Keys the code reads

| Key | Used for |
|---|---|
| `tokenizer.vocab_size.*` | `special` (4), `note` (128), `time` (must be ≥ steps per segment, e.g. 100), `duration` (default 32), `velocity_bins` (default 0 = no dynamics). The old `velocity` key is ignored |
| `t5.*` | Overrides on the `t5-small` config (`d_model`, `num_layers`, `vocab_size`, ...) |
| `dataset.n_bars` | Segment length: `n_bars * 4` beat steps |
| `dataset.target_length` | Maximum number of target tokens (also the base `max_length` at inference) |
| `dataset.mel_is_conditioned` | Composer conditioning on/off (default `true`) |
| `dataset.sample_rate` | Only the sample rate of the rendered mix at inference. Ignored for training |
| `composer_to_feature_token` | Composer name → token value |
| `training.lr` | Base learning rate |
| `training.optimizer` | `adafactor` or anything else for AdamW (required key) |
| `training.batch_size`, `num_gpu`, `max_epochs`, `accumulate_grad_batches`, `gradient_clip_val`, `check_val_every_n_epoch` | Trainer settings |

`dataset.use_mel` is no longer used.

### New optional keys (defaults shown)

```yaml
encoder:
  name: m-a-p/MERT-v1-95M   # pretrained audio encoder
  revision: null            # pin a commit hash (trust_remote_code runs hub code)
  sample_rate: 24000        # must match the encoder
  freeze: true              # false = fine-tune MERT
  normalize_waveform: true
  context_seconds: 2.0      # audio taken on each side of a segment
  lr_mult: 0.1              # MERT LR multiplier when fine-tuning

training:
  warmup_steps: 500
  weight_decay: 0.01        # AdamW only
  scheduler: cosine         # or "none"
  log_every_n_steps: 50
```

### Command line

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py DATA_DIR [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `config.yaml` | Config file |
| `--composer` | `composer1` | Key in `composer_to_feature_token` |
| `--output_dir` | `checkpoints` | Lightning root dir |
| `--num_workers` | 4 | DataLoader workers (audio decoding is the bottleneck) |
| `--max_epochs` | config value | Overrides `training.max_epochs` |
| `--validation_fraction` | 0.1 | Fraction of **songs** used for validation |
| `--train_stride_fraction` | 0.5 | Segment stride for training, as a fraction of the segment length |
| `--precision` | Lightning default | For example `bf16-mixed` |
| `--resume` | none | Checkpoint to resume from |
| `--seed` | 3407 | Seed for training and for the split |

---

## 6. Training/inference consistency

Inference (`TransformerWrapper.generate`) reuses the same input path, so these must match what the model saw during training:

- **`steps_per_beat`** (2 in `generate`) and the way the beat grid is built.
- **`n_bars`**: onset positions are **absolute** inside a chunk, so inference must use the same `dataset.n_bars` as training (the default). Other values print a warning and may generate positions never seen in training.
- **Context**: inference uses full `context_seconds` on both sides, while training chunks may be truncated at their edges.
- **Encoder settings**: `encoder.name`, `sample_rate`, `normalize_waveform` must be the same.
- **Composer**: pass a composer from `composer_to_feature_token`.

Inference-only options: sampling (`generation_kwargs`, `seed`). Chunks are generated independently and simply concatenated: each one only writes the notes that start in it, so no overlap or stitching is needed.

---

## 7. Practical notes and limitations

- **Old checkpoints are incompatible**: the encoder input changed (MERT + beat embedding instead of log-mel).
- **First run downloads** `t5-small` (config) and the MERT weights from the Hugging Face hub.
- **Context is bounded by chunk length** in training. If you want more, increase the cleaner's `--window-seconds`.
- **No downbeat information**: beat positions are relative to the segment start.
- **Suggested progression**: train with MERT frozen first, then (optionally) unfreeze with `lr_mult: 0.1` and a lower base LR if validation loss plateaus.
- **Only `val_loss` is monitored.** It measures token prediction, not musical quality. Listen to generated samples, or compute alignment metrics such as the cleaner's chroma similarity on generated MIDI, before drawing conclusions.
- **Memory**: the layer-weighted sum keeps all MERT hidden states for the backward pass. If you run out of memory, lower `training.batch_size` and increase `accumulate_grad_batches`.