# Installation:

2 environments are required for the data:

- The first to fetch youtube:

```bash
conda create -n ytdlp python=3.11 -y
conda activate ytdlp
python -m pip install -U "yt-dlp[default]"
conda install -c conda-forge ffmpeg nodejs -y
python -m pip install pandas tqdm joblib omegaconf
```

```bash
# test with --num_audio 1 to download the first row of the dataset
python download/download.py     train_dataset.csv     data     --num_audio 1 
```

Each piano/song pair is stored in its own numbered directory
(`data/data_0`, `data/data_1`, ...). The downloader uses the active Python
environment's `yt-dlp` module and ignores incomplete `.part` files. If a pair
cannot be downloaded after the configured retries, that pair is skipped and
the remaining pairs continue.

- The second to run the following scripts:

```bash
conda create -n music2piano python=3.9
pip install -r requirements.txt
# rubberband-cli is also required by pop_align.py
conda install -c conda-forge rubberband ffmpeg
```

`pop_align.py` uses `pyrubberband`, which is only the Python wrapper; it also
needs the external `rubberband` executable. If it is missing, alignment stops
before processing samples so later stages are not run on incomplete files.


# Preprocess Scripts

1. transform wav to midi 
```bash
python wav_to_midi.py DATA_DIR
```

2. synchronize midi 
```bash
python pop_align.py DATA_DIR
```

3. Estimate Pop's beats
```bash
python bpm_quantize.py DATA_DIR 
```

4. get separated vocal track
```bash
python split_spleeter.py DATA_DIR
```

5. caculate melody chroma accuracy
```bash
python melody_accuracy.py DATA_DIR 
```

6. keep only well-aligned chunks
```bash
python clean_aligned_data.py DATA_DIR data_clean --threshold 0.35
```


full pipeline:
```bash
python preprocess/wav_to_midi.py data
python preprocess/pop_align.py data
python preprocess/bpm_quantize.py data 
python preprocess/split_spleeter.py data
python preprocess/melody_accuracy.py data 
python preprocess/clean_aligned_data.py data data_clean 

```
`clean_aligned_data.py` evaluates overlapping beat-aligned windows by comparing
the full-song chroma with the MIDI chroma. This is the primary acceptance
metric because the MIDI is a piano accompaniment, not necessarily the vocal
melody. Vocal melody metrics are retained as diagnostics only, so good
piano/song alignment is not discarded merely because the piano's highest note
does not match the singer. It keeps windows whose local
`music_chroma_similarity` is at least the selected threshold and writes each
window as an independent sample under `data_clean/data_clean_i/`, with one clean numbered directory corresponding to
each source song directory `data/data_i/`. The output has the same
metadata-at-root and song-directory layout as its source, including
`*.pitchshift.wav`, `*.qmidi.mid`, `*.qmix.flac`, `*.beatstep.npy`, and
`*.notes.npy`, so it can be passed directly to the training script or
inspected as a MIDI/song stereo mix. The root-level `<chunk>.wav` is also the
chunked song audio, not the original full piano recording. Chunks are also
checked for audio/MIDI activity overlap and matched note-onset timing, preventing a
high pitch/chroma score alone from accepting a badly timed section. Each
chunk YAML records the music chroma similarity and coverage, vocal melody
diagnostics, voicing, activity IoU, activity ratio, onset F1, note-density
ratio, synchronized/null chroma, and alignment margin metrics plus its source
time interval. The alignment margin is synchronized chroma similarity minus a
time-shuffled MIDI-chroma baseline, preventing a globally similar key or chord
vocabulary from passing without temporal agreement. Onset F1 is diagnostic by
default and is not used as a hard filter unless `--min-onset-f1` is explicitly
set.

Useful options:

```bash
python clean_aligned_data.py DATA_DIR data_clean \
  --threshold 0.40 \
  --window-seconds 8 \
  --hop-seconds 4 \
  --min-notes 12 \
  --min-note-onsets 4 \
  --min-activity-iou 0.20 \
  --min-alignment-margin 0.05 \
  --min-onset-f1 0.05
```

Sparse piano sections are filtered per chunk, not per complete song. By
default a retained chunk must contain at least 12 MIDI notes and 4 distinct
MIDI onset times. These values can be changed with `--min-notes` and
`--min-note-onsets`; the actual counts are stored as `eval.midi_note_count`
and `eval.midi_onset_count` in each chunk YAML.

The script refuses to overwrite a non-empty output directory unless
`--overwrite` is supplied.

For a storage-efficient training dataset, use:

```bash
python clean_aligned_data.py data data_clean --minimal --overwrite
```

`--minimal` sets the hop equal to the window length, so consecutive windows
do not overlap. It writes only the metadata, chunked `*.pitchshift.wav`,
`*.qmidi.mid`, `*.beatstep.npy`, and `*.notes.npy` files needed by the
training loader and MIDI target pipeline. It omits qmix audio, duplicate
root-level WAV/MIDI files, duplicate `.mid` files, and auxiliary beat arrays.

# Expected Structure
```
├── -7lV0oJ0QXc
│   ├── EHl_eQhgefw.beatinterval.npy
│   ├── EHl_eQhgefw.beatstep.npy
│   ├── EHl_eQhgefw.beattime.npy
│   ├── EHl_eQhgefw.mid
│   ├── EHl_eQhgefw.notes.npy
│   ├── EHl_eQhgefw.pitchshift.wav
│   ├── EHl_eQhgefw.qmidi.mid
│   ├── EHl_eQhgefw.qmix.flac
│   ├── EHl_eQhgefw.vocals.mp3
│   ├── EHl_eQhgefw.wav
│   └── The Beatles - With a Little Help from My Friends ____With A Little Help From My Friends - The Beatles _.txt
├── -7lV0oJ0QXc.mid
├── -7lV0oJ0QXc.wav
├── -7lV0oJ0QXc.yaml
```

## Descriptions for each data
1. ```*.beattime.npy```
    - timesteps (unit : second) extracted using essentia. ```np.ndarray```. (num_beats, )
2. ```*.beatstep.npy```
    - timesteps (unit : second) per every half-beat. it is calculated using linear interpolation of ```beattime```.
2. ```*.notes.npy```
    - ```np.ndarray``` shape: ```(number_of_notes, 4)```
    - each row contains : ```[onset(unit: index), offset(unit: index), pitch, velocity]```
    - onset/offset values mean that the index of ```beatstep``` time. 
    - for example, 
        - ```beatstep = [0.6, 1.0, 1.4]```
        - ```note = [0, 1, 77, 88]``
        - then ```note``` means a note starts from 0.6sec to 1.0sec, and its pitch is 77 and velocity is 88.
