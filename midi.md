# MIDI data flow

This document explains the two classes used to prepare and handle MIDI-related
data in this project:

- [`MidiAudioPair`](./midiaudiopair.py) manages the relationship between a
  dataset metadata file and the audio, MIDI, and intermediate files belonging
  to one song.
- [`MidiTokenizer`](./midi_tokenizer.py) converts quantized piano notes into
  integer sequences for the model and decodes those sequences back into notes
  or a MIDI file.

The classes solve different problems. `MidiAudioPair` is responsible for
dataset organization and validation; `MidiTokenizer` is responsible for the
representation used by the sequence model.

## 1. `MidiAudioPair`

### Purpose

A `MidiAudioPair` represents one piano performance paired with its source song.
It is initialized from a YAML metadata file. The object:

1. Loads the metadata with OmegaConf.
2. Determines the directory containing the song assets.
3. Validates the metadata against the dataset rules.
4. Constructs paths for the expected source and generated files.
5. Checks which expected files are present.
6. Exposes an `is_valid` flag and an error code for preprocessing and
   evaluation scripts.

### Creating a pair

```python
from midiaudiopair import MidiAudioPair

sample = MidiAudioPair(
    "metadata/example.yaml",
    audio_dir="data/audio",
)

if sample.is_valid:
    print(sample.midi)
else:
    print(sample.error_code, sample.invalids)
```

If `audio_dir` is omitted, the directory containing the YAML file is used.
The metadata is expected to contain at least `piano` information, and normally
also a `song`, `tempo`, and evaluation section.

### Validation

`validate_yaml` returns one of the class error codes. The checks include:

| Code | Meaning |
| --- | --- |
| `VALID` | Metadata passes all metadata checks. |
| `NO_SONG` | The YAML does not contain a song entry. |
| `NO_PIANO` | The YAML does not contain a piano entry. |
| `NO_SONG_DIR` | The directory for the piano recording does not exist. |
| `BAD_DURATION` | Piano and song durations differ by too much. |
| `BAD_TITLE` | The piano title matches the excluded title pattern. |
| `NO_TEMPO` | Tempo information is missing. |
| `BLACKLIST` | The piano YouTube ID is blacklisted. |
| `BAD_ACCURACY` | Melody chroma accuracy is below the training-data threshold. |

After metadata validation, `validate_files` checks the expected generated
assets. Missing paths are returned in `sample.invalids`; the object is valid
only when the metadata code is `VALID` and this list is empty.

```python
missing_files = sample.validate_files()
if missing_files:
    print("Missing:", missing_files)
```

### Paths exposed by the object

For a valid song, `set_song_attrs` creates paths based on the song ID. Common
attributes include:

- `midi`: source or processed MIDI file
- `song`: pitch-shifted song audio
- `mix`: mixed audio
- `beattime`: beat times in seconds
- `beatstep`: quantized beat positions used by the tokenizer
- `beatinterval`: beat intervals
- `qmidi`: quantized MIDI
- `qmix`: quantized audio mix
- `notes`: NumPy note-event array
- `vqvae`: VQ-VAE features
- `vocals`: separated vocals

The `generated` and `result_json` methods return paths for model outputs:

```python
generated_midi = sample.generated(
    composer="pop2piano",
    generated="generated",
)
result_file = sample.result_json(generated="generated")
```

## 2. Musical meaning and relationships

These files are different representations of one paired example: a piano
cover and its corresponding pop song.

```text
piano-cover audio --> piano transcription --> midi

pop-song audio --> alignment/pitch correction --> song
                         |
                         +--> beat tracking --> beattime
                         |                    beatinterval
                         |                    beatstep
                         |
                         +--> quantized MIDI/audio --> qmidi, qmix, notes
```

For a song ID such as `EHl_eQhgefw`, the derived files are stored below the
piano directory:

```text
data/<piano_id>/EHl_eQhgefw.<suffix>
```

### `midi`: source or aligned piano MIDI

This is a symbolic score of the piano cover. It stores pitches, note-on and
note-off times, velocities, and possibly sustain-pedal events; it is not audio.

The original transcription is normally:

```text
data/<piano_id>.mid
```

`pop_align.py` reads this MIDI and writes the time-aligned result as
`data/<piano_id>/<song_id>.mid`. The note pitches are retained; the note times
are changed to follow the pop-song timeline. If `W` is the estimated
time-warping function, a piano note `[p_start, p_end]` becomes:

```text
aligned_start = W(p_start)
aligned_end   = W(p_end)
```

This makes the piano performance line up with the pop song. Thus the
processed MIDI is changed, but the original MIDI remains available at the
top-level path.

### `song`: pitch-shifted pop-song audio

This is a derived copy of the original pop recording after preprocessing. The
downloaded source is preserved separately as:

```text
data/<piano_id>/<song_id>.wav
```

The derived `song` is pitch-shifted so that the pop audio and the piano MIDI
use the same musical key. It is not time-stretched: the MIDI time axis is
warped instead. If the estimated key correction is `s` semitones, a frequency
`f` is shifted according to:

```text
f_corrected = f * 2 ** (s / 12)
```

The current `pop_align.py` writes the derived file at 22,050 Hz, and the
training loop also loads it at 22,050 Hz. Pitch shifting changes frequency
(and therefore musical key), but does not change beat timestamps.

### `mix`: aligned piano-and-song audio

This is intended to be a listening preview combining the song with a
FluidSynth rendering of the aligned piano MIDI:

```text
mix(t) = song(t) + alpha * rendered_piano(t)
```

The path is declared by `MidiAudioPair`, but the current preprocessing code
primarily writes `qmix` rather than `.mix.flac`; therefore `mix` may not exist
in every dataset checkout.

### Which signal is preserved and which is changed?

The pipeline does not replace the original pop audio. It keeps the downloaded
file and creates processed derivatives:

```text
original pop audio
    +-- preserved: data/<piano_id>/<song_id>.wav
    |
    +-- pitch-shifted copy: data/<piano_id>/<song_id>.pitchshift.wav

original piano MIDI transcription
    +-- preserved: data/<piano_id>.mid
    |
    +-- time-warped copy: data/<piano_id>/<song_id>.mid
```

The alignment procedure is:

1. Render the original piano MIDI to audio.
2. Compare piano-rendered audio and original pop audio using chroma/onset
   features.
3. Estimate a time map `W` with MrMsDTW.
4. Apply `W` to the piano MIDI note times.
5. Estimate a chromatic key difference.
6. Apply the opposite pitch shift to the derived pop audio.

The reason for changing both representations in different ways is that the
training pair must agree in both dimensions:

```text
time:  aligned MIDI time ~= pop-song time
key:   MIDI pitches ~= pitch-shifted pop-song frequencies
```

It would be possible to keep the pop audio unchanged and transpose the MIDI
instead, but that is not what this repository implements. Transposing the
MIDI would change the piano arrangement's written pitches, while the current
pipeline preserves those pitches and normalizes the audio key instead.

### `beattime`: detected beat times in seconds

This is an array:

```text
beattime = [t_0, t_1, ..., t_(B-1)]
```

where each `t_i` is a detected beat position measured from the start of the
song. Neighboring values define the local beat duration and tempo:

```text
beat_duration_i = t_(i+1) - t_i
tempo_BPM_i     = 60 / beat_duration_i
```

The values are estimated by Essentia's `RhythmExtractor2013` from the
pitch-shifted `song` file. Because pitch shifting does not alter the time
axis, these beat times correspond to the original pop recording's timing.

### `beatinterval`: beat durations

This stores Essentia's beat-interval output. In a regular beat track it is
approximately:

```text
beatinterval[i] = beattime[i+1] - beattime[i]
```

The local tempo is therefore approximately:

```text
local_BPM[i] = 60 / beatinterval[i]
```

Because this is the tracker output, its boundary convention and length may
differ from `beattime`; do not assume the two arrays have identical shapes.

### `beatstep`: quantized eighth-note time grid

This is the time grid used to convert continuous note times into integer
positions. With two steps per beat, neighboring detected beats `t_i` and
`t_(i+1)` produce approximately:

```text
u_(2i)   = t_i
u_(2i+1) = (t_i + t_(i+1)) / 2
u_(2i+2) = t_(i+1)
```

The values `u_i` are still seconds, but their indices are used by the model.
For a note row `[onset_idx, offset_idx, pitch, velocity]`:

```text
start_time = beatstep[onset_idx]
end_time   = beatstep[offset_idx]
duration   = end_time - start_time
```

The quantizer assigns each continuous onset and offset to the nearest grid
position:

```text
onset_idx  = nearest_index(beatstep, onset_seconds)
offset_idx = nearest_index(beatstep, offset_seconds)
```

### `qmidi`: quantized piano MIDI

This is a normal MIDI file whose note times have been snapped to `beatstep`.
For every aligned piano note:

```text
quantized_start = beatstep[onset_idx]
quantized_end   = beatstep[offset_idx]
```

`qmidi` is the renderable form of the same quantized score represented by
`notes.npy`.

### `qmix`: mix using quantized MIDI

This is a listening preview created by rendering `qmidi` with FluidSynth and
mixing it with `song`:

```text
qmidi --> FluidSynth --> piano waveform
song  -----------------> song waveform
both  --> get_stereo --> qmix.flac
```

It is useful for checking alignment and quantization, but it is not the main
audio input to the transformer.

### `notes`: NumPy note-event array

This is the compact score used by the tokenizer. Each row is:

```text
[onset_idx, offset_idx, pitch, velocity]
```

| Column | Musical meaning |
| --- | --- |
| `onset_idx` | Start index in `beatstep` |
| `offset_idx` | End index in `beatstep` |
| `pitch` | MIDI pitch; 60 is middle C (C4) |
| `velocity` | MIDI attack strength, normally 1–127 |

For example:

```text
beatstep = [0.60, 0.95, 1.30, 1.65]
notes[0] = [0, 2, 60, 90]
```

means “play C4 with velocity 90 from 0.60 s to 1.30 s.” `train.py` clips
these rows to local windows and converts them into relative time, velocity,
and pitch tokens.

### `vqvae`: optional VQ-VAE audio features

This is intended to contain discrete audio codes produced by a VQ-VAE
encoder. Such codes can replace waveform/mel input in an alternative model.
The current configuration uses:

```yaml
dataset:
  use_mel: true
```

and the current repository does not create `.vqvae.pt`, so this file is not
required by the current mel-based training or inference path.

### `vocals`: separated vocal track

This is the vocal stem extracted from the pop-song mix by Spleeter. It is
mainly used for quality evaluation:

```text
vocals --> f0(t)
qmidi  --> highest piano pitch(t)
both   --> melody chroma/pitch accuracy
```

The resulting accuracy is stored in the metadata and used to reject poor
pairs. `vocals` is not the principal transformer input and is not a MIDI
representation.

### Melody accuracy fields and training-data threshold

The `eval` section of the metadata stores two complementary comparisons
between the melody estimated from the vocal stem and the melody represented by
the quantized piano MIDI:

```yaml
eval:
  melody_chroma_accuracy: 0.22377447923163277
  melody_pitch_accuracy: 0.014593987775976052
```

Both values are returned by `mir_eval.melody` and are in the range `0.0` to
`1.0`, where larger values indicate better agreement.

- **`melody_chroma_accuracy`** measures whether the estimated and reference
  melodies agree in pitch class (the note name modulo octaves). For example,
  C3 and C4 have the same chroma. This makes the metric tolerant of octave
  errors while still checking whether the melody follows the same harmonic
  pitch classes.
- **`melody_pitch_accuracy`** measures whether the estimated and reference
  melodies agree in their actual pitch, including octave. It is therefore
  stricter than chroma accuracy: C3 and C4 do not count as the same pitch.

The default threshold used by `MidiAudioPair` for accepting a sample as
training data is:

```text
melody_chroma_accuracy >= 0.15
```

The check is implemented in `MidiAudioPair.validate_yaml`. A value below
`0.15`, a missing `eval` section, or a missing
`melody_chroma_accuracy` field produces `BAD_ACCURACY`, so the pair is not
marked as valid training data. The pitch metric is currently recorded for
diagnostics and evaluation only; **there is no independent default
`melody_pitch_accuracy` threshold in the current code**.

For the example above, `0.2238 >= 0.15`, so the sample passes the accuracy
criterion despite its low pitch accuracy value. To introduce a pitch-based
filter, `validate_yaml` would need an additional explicit threshold check.

### Cleanup

`delete_files_myself` removes the song directory, YAML metadata, original MIDI,
and original WAV file. It is destructive and is primarily used with
`auto_remove_no_song=True` when orphaned metadata is detected:

```python
sample = MidiAudioPair(
    "data/orphan.yaml",
    audio_dir="data/audio",
    auto_remove_no_song=True,
)
```

## 3. `MidiTokenizer`

### Purpose

`MidiTokenizer` translates between quantized piano notes and the integer
vocabulary consumed by the sequence model. It uses the vocabulary sizes in the
configuration:

```python
tokenizer = MidiTokenizer(config)
```

The tokenizer expects note events in the following format:

```text
(onset_idx, offset_idx, pitch, velocity)
```

For example:

```python
notes = np.array([
    [0, 4, 60, 90],  # C4 starts at beat index 0 and ends at index 4
    [4, 8, 64, 80],  # E4 starts at beat index 4 and ends at index 8
])
```

`onset_idx` and `offset_idx` refer to entries in a quantized beat grid, not
seconds. `pitch` is a MIDI pitch and `velocity` is the MIDI velocity. In the
current token representation, velocity is reduced to a binary state:
`velocity > 0` means note-on and `velocity == 0` means note-off.

## 4. Token vocabulary

The token ID ranges are contiguous:

```text
special tokens:  0 ... n_special - 1
note tokens:     n_special ... n_special + n_note - 1
velocity tokens: n_special + n_note ... n_special + n_note + n_velocity - 1
time tokens:     after the velocity range
```

The module defines four token types:

| Constant | Meaning |
| --- | --- |
| `TOKEN_SPECIAL` | Padding, end-of-sequence, tie, or another special value. |
| `TOKEN_NOTE` | A MIDI pitch. |
| `TOKEN_VELOCITY` | A velocity state, currently normally `0` or `1`. |
| `TOKEN_TIME` | A beat index or a relative time shift. |

`fast_tokenize` performs the numerical offset calculation and
`fast_detokenize` reverses it. `tokenize_note` and `detokenize` are the
configuration-aware methods normally used by application code.

```python
from midi_tokenizer import TOKEN_NOTE

token = tokenizer.tokenize_note(60, TOKEN_NOTE)
token_type, value = tokenizer.detokenize(token, time_idx_offset=0)
```

## 5. Absolute versus relative tokens

### Absolute tokens

`notes_to_tokens` emits a time token containing the absolute beat index for
each occupied time position. It is useful when the sequence describes a
complete piece or another sequence with a fixed global origin:

```python
absolute_tokens = tokenizer.notes_to_tokens(notes)
print(tokenizer.to_string(absolute_tokens))
```

At each occupied beat, the sequence contains:

1. A time token.
2. A velocity-state token when the state changes.
3. One note token for every note event at that time.

Both the onset and offset of a note are represented as note events. An onset
uses a positive velocity and an offset uses zero velocity.

### Relative tokens

`notes_to_relative_tokens` uses `offset_idx` as time zero. Instead of absolute
beat positions, time tokens contain the number of beats since the previous
event. Relative sequences are useful for fixed-size training windows because
the same local pattern can be represented independently of its position in the
full song.

```python
relative_tokens = tokenizer.notes_to_relative_tokens(
    notes,
    offset_idx=0,
    add_eos=True,
)
```

The optional `add_eos` flag appends the `EOS` special token. The optional
`add_composer` flag prefixes a composer token, which allows a model to receive
composer/style conditioning as part of the sequence.

## 6. Decoding model output

`relative_tokens_to_notes` reads a relative sequence and reconstructs absolute
note-event rows. It tracks:

- the current beat index,
- the current velocity state,
- the active onset for each pitch.

When a note-off event is encountered, the stored onset and current index become
one `(onset_idx, offset_idx, pitch, velocity)` row. Notes that remain active at
the end of a sequence receive a forced offset so the decoded result is
complete.

```python
decoded_notes = tokenizer.relative_tokens_to_notes(
    relative_tokens,
    start_idx=0,
)
```

`relative_batch_tokens_to_midi` performs this operation for a batch of windows.
It offsets each window according to its position and joins the decoded notes
before converting them into a MIDI object.

```python
midi, decoded_notes = tokenizer.relative_batch_tokens_to_midi(
    token_batch,
    beatstep,
    beat_offset_idx=0,
    bars_per_batch=2,
)
midi.write("generated.mid")
```

## 7. Converting notes to MIDI

`notes_to_midi` maps note indices to seconds using `beatstep`. For each note,
the start time is `beatstep[onset_idx]` and the end time is
`beatstep[offset_idx]`. The optional `offset_sec` subtracts a common time
origin, which is useful when exporting a segment that starts partway through a
song.

```python
midi = tokenizer.notes_to_midi(
    decoded_notes,
    beatstep,
    offset_sec=beatstep[0],
)
midi.write("segment.mid")
```

The resulting `PrettyMIDI` object contains one piano instrument with program
number `0`. Invalid notes are removed before the object is returned.

## 8. Selecting a song segment

`split_notes` selects notes whose onset is between two times. It converts the
requested seconds to beat indices using `np.searchsorted`, then slices the
note array. The notes must be sorted by onset, and `beatsteps` must be sorted
by time.

```python
segment, (start_beat, end_beat, start_note, end_note) = (
    tokenizer.split_notes(notes, beatsteps, time_from=16.0, time_to=32.0)
)

segment_tokens = tokenizer.notes_to_relative_tokens(
    segment,
    offset_idx=start_beat,
    add_eos=True,
)
```

This is the typical segment pipeline:

```text
metadata YAML
      |
      v
MidiAudioPair          -> validate files and locate beatstep/notes
      |
      v
quantized notes        -> split into a training or inference window
      |
      v
MidiTokenizer          -> convert notes to relative tokens
      |
      v
model                  -> generate/predict token sequences
      |
      v
MidiTokenizer          -> decode tokens to notes and PrettyMIDI
      |
      v
generated MIDI file
```

## 9. Practical considerations

- Use quantized beat indices consistently. Do not pass seconds directly as
  `onset_idx` or `offset_idx`.
- Keep notes sorted by onset before calling `split_notes`.
- `beatstep` must contain entries for every note onset and offset used during
  MIDI export.
- Relative token sequences need the correct `start_idx` when decoded; otherwise
  notes will be placed at the wrong position in the song.
- `EOS`, `PAD`, and `TIE` are special token values. `to_string` makes these
  values easier to inspect while debugging model inputs and outputs.
- `extrapolate_beat_times` can extend a beat-time array when an export needs
  beat positions beyond the measured sequence.
