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
| `BAD_ACCURACY` | Melody/chroma accuracy is below the minimum threshold. |

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

## 2. `MidiTokenizer`

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

## 3. Token vocabulary

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

## 4. Absolute versus relative tokens

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

## 5. Decoding model output

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

## 6. Converting notes to MIDI

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

## 7. Selecting a song segment

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

## 8. Practical considerations

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
