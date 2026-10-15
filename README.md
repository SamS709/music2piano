# Music2Piano : Pop Audio-based Piano Cover Generation

This project generates piano MIDI arrangements from pop audio. The repository
contains the model and preprocessing code. The training entry point is
`train.py`; it is a reconstructed training loop because the original training
script is not included in this checkout.

## How to prepare dataset
### Download Original Media
---
- List of data : ```train_dataset.csv```
- Downloader : ```download/download.py```
    - ```python download.py train_dataset.csv data/```

### Preprocess Data
---
- [Details](./preprocess/)

The training dataset must be fully preprocessed before training. For each
metadata file in `data/`, the corresponding song directory must contain at
least:

```text
<song_id>.pitchshift.wav
<song_id>.beatstep.npy
<song_id>.notes.npy
```

Run the preprocessing scripts in the documented order after creating the
initial piano MIDI:

```bash
python preprocess/pop_align.py data
python preprocess/bpm_quantize.py data
python preprocess/split_spleeter.py data
python preprocess/melody_accuracy.py data
```

## Training environment

The original `requirements.txt` is intended for the legacy preprocessing
environment. Training uses a separate environment defined in
`requirements_train.txt`. This is recommended for modern NVIDIA GPUs,
including RTX 50-series cards.

Create and activate the environment:

```bash
conda create -n pop2piano-train python=3.11
conda activate pop2piano-train
```

Install the training dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_train.txt
```

The training requirements install PyTorch 2.7.1 with CUDA 12.8 wheels. Verify
that PyTorch can use the GPU before starting a long run:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

## Launch training

From the project root, run a one-epoch smoke test first:

```bash
python train.py \
    data \
    --config config.yaml \
    --composer composer1 \
    --max_epochs 1 \
    --num_workers 0 \
    --output_dir checkpoints_smoke
```

For a larger dataset, use several data-loader workers and a longer run:

```bash
python train.py \
    data \
    --config config.yaml \
    --composer composer1 \
    --max_epochs 100 \
    --num_workers 4 \
    --output_dir checkpoints
```

Useful arguments are:

| Argument | Meaning |
| --- | --- |
| `data` | Directory containing top-level `*.yaml` metadata files |
| `--config` | Model and training configuration; defaults to `config.yaml` |
| `--composer` | Composer token from `config.yaml`, for example `composer1` |
| `--max_epochs` | Number of training epochs |
| `--num_workers` | DataLoader worker processes; use `0` for debugging |
| `--validation_fraction` | Fraction of metadata files used for validation |
| `--output_dir` | Lightning logs and training output directory |
| `--seed` | Reproducibility seed |

The current script does not include the original dataset split, augmentation,
or arranger-label metadata. It uses one selected composer token for all
examples. The downloaded metadata does not identify the original arranger, so
this is suitable for validating and training the reconstructed loop, but it is
not an exact reproduction of the original 21-arranger experiment.

## How the training loop works

`train.py` builds examples from the preprocessed files:

1. `Pop2PianoDataset` reads each top-level metadata file and keeps samples
   containing `*.pitchshift.wav`, `*.beatstep.npy`, and `*.notes.npy`.
2. Each song is divided into `n_bars * 4` eighth-note steps. With the default
   configuration (`n_bars: 2`), one example contains two bars.
3. The matching audio interval is loaded at 22,050 Hz and converted by the
   existing `LogMelSpectrogram` module (`n_fft=4096`, `hop_length=1024`,
   `n_mels=512`).
4. Quantized notes overlapping the interval are clipped to the segment and
   converted to relative MIDI tokens with `MidiTokenizer`. An EOS token is
   appended.
5. `collate_batch` pads audio with zeros and target tokens with `-100`.
   Hugging Face T5 ignores `-100` positions when calculating cross-entropy.
6. `TrainableTransformer` optionally prepends the selected composer embedding
   using `ConcatEmbeddingToMel`, then passes the mel features and labels to
   `T5ForConditionalGeneration`.
7. PyTorch Lightning calls `training_step`, logs `train_loss`, and updates the
   model with Adafactor (the optimizer selected by `config.yaml`).
8. `validation_step` calculates and logs `val_loss`. The metadata files are
   shuffled deterministically and split according to `--validation_fraction`.

The model learns to predict MIDI tokens from audio during training. At
inference, the existing generation code in
`evaluate/inference_testset.py` autoregressively decodes those tokens and
writes standard MIDI files.

## Training output and limitations

The smoke test should complete with output similar to:

```text
GPU available: True (cuda), used: True
Epoch 0: ... train_loss=...
`Trainer.fit` stopped: `max_epochs=1` reached.
```

Lightning logs are written below the selected `--output_dir`. The current
reconstructed script does not explicitly configure a `ModelCheckpoint`
callback, so inspect the output directory after training rather than assuming
that a `.ckpt` file was saved:

```bash
find checkpoints_smoke -type f | sort
```

One processed song is enough for a smoke test, but not for useful learning.
Meaningful training requires many independent processed songs and a proper
arranger/composer label for each example if style conditioning is required.



