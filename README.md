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
    - ```python download/download.py train_dataset.csv data/ --num_audio 4000```
    - Use `--start_audio 3000` to skip the first 3000 CSV rows. Combine it
      with `--num_audio` to select a batch, for example
      `--start_audio 3000 --num_audio 1000`.

The downloader uses no intentional delay by default. If YouTube starts
rate-limiting requests, add delays such as
`--sleep-requests 3 --sleep-min 10 --sleep-max 20
--sleep-between-pairs 30`.

Each downloaded piano/song pair gets its own numbered dataset directory below
`data/`, such as `data/data_0` and `data/data_1`. Training and preprocessing
commands accept the container directory (`data`) and discover all numbered
datasets automatically.

### Preprocess Data
---
- [Details](./preprocess/)

The training dataset must be fully preprocessed before training. For each
metadata file in `data/data_i/`, the corresponding song directory must contain at
least:

```text
<song_id>.pitchshift.wav
<song_id>.beatstep.npy
<song_id>.notes.npy
```

Run the preprocessing scripts in the documented order after creating the
initial piano MIDI:

```bash
python preprocess/wav_to_midi.py data
python preprocess/pop_align.py data
python preprocess/bpm_quantize.py data
python preprocess/split_spleeter.py data
python preprocess/melody_accuracy.py data
python preprocess/cleaner.py data
```

Spleeter uses TensorFlow and will use a CUDA GPU automatically when the
TensorFlow CUDA dependencies are installed. Check the selected device before
processing the full dataset:

```bash
python preprocess/split_spleeter.py data --device gpu
```

Use `--device cpu` to force CPU processing, or omit the option to allow
automatic CPU fallback when TensorFlow cannot see a GPU.

For a separate Spleeter environment, avoid installing the full legacy
requirements file because it contains unrelated packages with incompatible
historical NumPy constraints:

```bash
conda create -n music2piano-gpu python=3.10 -y
conda activate music2piano-gpu
python -m pip install -r requirements_spleeter_gpu.txt
python preprocess/split_spleeter.py data --device gpu
```

The focused requirements file includes the CUDA 11.8/cuDNN 8.6 runtime wheels
needed by TensorFlow 2.12.1. In WSL, set the native library path before
starting Python (the dynamic loader reads it before TensorFlow is imported):

```bash
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$(find "$CONDA_PREFIX/lib/python3.10/site-packages/nvidia" -type d -name lib -printf '%p:')${LD_LIBRARY_PATH}"
python preprocess/split_spleeter.py data --device gpu
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

The main entry point is `train.py`. The script builds a deterministic song-level
validation split, creates MERT/T5 training and validation loaders, then starts
PyTorch Lightning:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py data \
  --config config.yaml \
  --data-weight
```

`data` is the container directory containing the top-level metadata files and
processed dataset directories. Training is logged with Lightning CSV logging
and checkpoints are saved by `ModelCheckpoint`.

| Argument | Meaning |
| --- | --- |
| `data_dir` | Dataset container directory |
| `--config` | OmegaConf model configuration; default `config.yaml` |
| `--composer` | Composer token used for training |
| `--output_dir` | Training log/checkpoint root; default `training.output_dir` |
| `--num_workers` | DataLoader worker count |
| `--max_epochs` | Override `training.max_epochs` |
| `--seed` | Override `training.seed` |
| `--validation_fraction` | Fraction of independent songs held out for validation |
| `--train_stride_fraction` | Training segment stride fraction; default `0.5` |
| `--precision` | Override Lightning precision |
| `--data-weight` | Weight training chunks by cleaner region score |
| `--resume` | Continue a Lightning checkpoint with optimizer and epoch state |
| `--early_stopping_patience` | Val-loss checks without improvement before stopping; `-1` disables it |
| `--early_stopping_min_delta` | Minimum val-loss improvement required |

### Training behavior

`Pop2PianoDataset` creates `n_bars * 4` eighth-note-step segments. It reads the
preprocessed aligned audio, beat grid and quantized notes, and converts each
segment to onset/pitch/duration T5 tokens. Training segments use a configurable
overlap; validation uses disjoint segments.

Files are split by song identity, not by individual chunk. Covers that share the
same source song are kept together, preventing chunk overlap from leaking
information into validation.

The default config uses MERT at 24 kHz as the audio encoder and T5-small for
sequence decoding. MERT hidden states are mixed through learned
`audio_encoder.layer_logits` weights and projected to T5. Beat-position features
and composer conditioning are included in the encoder inputs.

Training augmentations are enabled only for the training dataset:

- 50% of segments are randomly pitch-shifted from -3 to -1 or 1 to 3 semitones.
  Audio and target pitch tokens are shifted together.
- MERT feature augmentation applies time masking, 5% feature dropout and small
  Gaussian noise only in training mode.

These values are configurable under `augmentation.pitch_shift` and
`encoder.feature_*` in `config.yaml`.

AdamW separates 2-D parameters and 1-D parameters and uses no weight decay for
the latter. Trainable MERT parameters receive
`main_learning_rate * encoder.lr_mult`; the default multiplier is `0.1`.

## Resume training

Use `--resume` to continue full Lightning state, including the optimizer,
scheduler and epoch:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py data \
  --resume checkpoints/lightning_logs/version_11/checkpoints/epoch=22-val_loss=1.2075.ckpt \
  --output_dir checkpoints
```

Do not combine `--resume` with `--finetune`.

## Fine-tune a trained checkpoint

Fine-tuning loads weights and starts a **new** Lightning trainer state.
By default, MERT remains frozen and only T5/projection parameters are trained:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py data \
  --finetune checkpoints/lightning_logs/version_11/checkpoints/epoch=22-val_loss=1.2075.ckpt \
  --finetune-lr 0.000005 \
  --output_dir checkpoints_finetune \
  --max_epochs 10
```

Fine-tuning options:

| Argument | Meaning |
| --- | --- |
| `--finetune` | Source checkpoint for weight initialization |
| `--finetune-lr` | Main/T5 learning rate; default `training.finetune_lr` |
| `--freeze-encoder` | Keep all MERT parameters frozen |
| `--unfreeze-encoder` / `--train-encoder` | Train only the top MERT blocks |
| `--encoder-top-layers` | Number of top MERT blocks; default `2` |

To train MERT's top two encoder blocks with `lr_mult: 0.1`:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py data \
  --finetune checkpoints/lightning_logs/version_11/checkpoints/epoch=22-val_loss=1.2075.ckpt \
  --unfreeze-encoder \
  --encoder-top-layers 2 \
  --finetune-lr 0.000005 \
  --output_dir checkpoints_finetune_encoder \
  --max_epochs 10
```

The resulting parameter-group learning rates are:

- Main/T5/projection parameters: `5e-6`
- Trainable top MERT blocks: `5e-6 * 0.1 = 5e-7`

### Fine-tune output location

For a source checkpoint under a Lightning `version_N` directory, metrics and
checkpoints are registered together beneath the source checkpoint name:

```text
checkpoints/lightning_logs/version_11/
  checkpoints/
    epoch=22-val_loss=1.2075.ckpt
  epoch=22-val_loss=1.2075/
    finetuned/
      metrics.csv
      epoch=0-val_loss=1.0000.ckpt
      last.ckpt
```

The CSV logger uses `name=<checkpoint stem>` and `version=finetuned`, so the
logger path is the same `.../finetuned/` directory used by `ModelCheckpoint`.
For a source checkpoint outside a `version_N` directory, the fallback is:

```text
<output_dir>/lightning_logs/<checkpoint_stem>/finetuned/
```

## Decode generated MIDI

Generation uses `MidiGrammarLogitsProcessor` to mask invalid T5 tokens:

- Onsets are non-decreasing.
- Every pitch is immediately followed by a duration.
- Onset, pitch, duration, velocity and special-token values stay in range.

Generated sequences are sanitized before decoding so orphan pitches and
malformed tokens cannot create MIDI notes.

## Compare beat grids

Run generation against the stored/preprocessed beat grid and an Essentia grid,
then report `mir_eval` onset F1 and note F1:

```bash
python evaluate/beat_note_f1.py data \
  --ckpt checkpoints/lightning_logs/version_11/checkpoints/epoch=22-val_loss=1.2075.ckpt \
  --config config.yaml \
  --composer composer1 \
  --model-id beat_eval \
  --output beat_note_f1.csv
```

## Training output and validation

Standard Lightning runs log below `--output_dir`:

```text
checkpoints/lightning_logs/
  version_0/
    metrics.csv
    checkpoints/
      epoch=*-val_loss=*.ckpt
      last.ckpt
```

At startup, the dataset prints kept/discarded segment counts, target-length
quantiles and pitch-shift augmentation accounting. The expected percentages
are printed as:

```text
pitch-shift augmentation: 50.00% of samples are augmented
(~N of M samples; +50.00% effective data; 66.67% original / 33.33% augmented)
```

Summarize all standard and fine-tune runs with:

```bash
python log_report.py --logs checkpoints/lightning_logs --output training_report.csv
```

The fine-tune CSV and nested checkpoint runs are discovered recursively.
Validation is song-level, so overlapping training chunks do not leak into
validation. A single processed song can smoke-test the pipeline, but meaningful
training requires multiple independent processed songs.
