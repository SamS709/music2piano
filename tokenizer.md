# MIDI Tokenizer: Onset + Duration

This document describes the first tokenizer implementation: the onset-and-duration MIDI tokenizer in `Pasted text.txt`.

## 1. Purpose and representation

The tokenizer converts MIDI notes represented as beat-step indices into integer token IDs, and decodes those tokens back into note rows or a MIDI object.

Each input note is represented as:

```text
(onset_idx, offset_idx, pitch, velocity)
```

- `onset_idx`: note onset in absolute beat-step indices.
- `offset_idx`: note offset in absolute beat-step indices.
- `pitch`: MIDI pitch value.
- `velocity`: MIDI velocity value.

A segment is the half-open interval `[start, end)`. It owns notes whose **onset** falls inside that interval. Notes are not clipped to the segment boundary: a note that starts inside the segment can have an offset beyond `end`.

The token sequence has this structure:

```text
TIME(t) [VELOCITY(v)] PITCH(p) DURATION(d) PITCH(p) DURATION(d) ...
TIME(t') ...
EOS
```

The optional velocity token is emitted only when velocity binning is enabled, and then only when the bin changes. A `TIME` token is emitted once per distinct onset time. Notes at a shared onset are sorted by pitch.

The key design choice is that each note is represented by its onset, pitch, and explicit duration. There are no note-off events.

## 2. Token types and special IDs

The implementation defines these token-type constants:

| Constant | Value | Meaning |
|---|---:|---|
| `TOKEN_SPECIAL` | `0` | Special tokens |
| `TOKEN_NOTE` | `1` | Pitch |
| `TOKEN_VELOCITY` | `2` | Velocity bin |
| `TOKEN_TIME` | `3` | Onset position within the segment |
| `TOKEN_DURATION` | `4` | Note duration |

Special token values:

| Constant | Value | Meaning |
|---|---:|---|
| `PAD` | `0` | Padding |
| `EOS` | `1` | End of sequence |
| `TIE` | `2` | Retained for compatibility; unused by this tokenizer |

These values are special-token *values* within the special-token vocabulary. The corresponding token IDs are the special vocabulary base plus the value; the special base is zero in this implementation.

## 3. Vocabulary layout and configuration

The vocabulary is laid out in contiguous ranges in this order:

```text
[special][pitch][time][duration][velocity]
```

The tokenizer reads vocabulary sizes from `config.vocab_size`:

| Key | Default | Meaning |
|---|---:|---|
| `special` | `4` | Number of special tokens |
| `note` | `128` | Number of pitch values |
| `time` | `100` | Number of time values |
| `duration` | `32` | Number of duration values |
| `velocity_bins` | `0` | Number of velocity bins; `0` disables velocity tokens |

The old `velocity` configuration key is ignored. Use `velocity_bins` to configure velocity quantization.

For each token type, the tokenizer stores a base ID equal to the sum of the sizes of all preceding ranges. An encoded token ID is:

```text
token_id = base_id[token_type] + value
```

The value must satisfy `0 <= value < vocabulary_size_for_type`, or `tokenize_note` raises `ValueError`.

`vocab_total` is the sum of all configured vocabulary sizes. The model's output vocabulary must be at least this large.

### Token value ranges

- Pitch values must fit in `[0, n_note)`. With the default `note=128`, pitches 0–127 are supported.
- Time values must fit in `[0, n_time)`.
- Duration values stored in tokens are zero-based: an actual duration `D` is stored as `D - 1`, so the token value range represents durations 1 through `n_duration`.
- Velocity-bin values fit in `[0, n_velocity)` when velocity binning is enabled.

A segment whose length `end - start` exceeds `n_time` is rejected. The implementation checks segment length, not only the largest onset observed in the segment.

## 4. Encoding a segment: `segment_to_tokens`

Signature:

```python
segment_to_tokens(notes, start, end, add_eos=True)
```

### Inputs

- `notes`: rows `(onset_idx, offset_idx, pitch, velocity)` for the whole song. Notes are not expected to be clipped to this segment.
- `start`, `end`: segment bounds in absolute beat steps, defining `[start, end)`.
- `add_eos`: whether to append the special `EOS` token (default `True`).

### Step-by-step behavior

1. Convert `start` and `end` to integers.
2. If `end - start > n_time`, raise `ValueError`.
3. Convert `notes` to a NumPy array.
4. Select only notes satisfying `start <= onset_idx < end`. Selection is based on onset only; a note's offset can be before or after the segment boundary.
5. Ignore selected notes whose pitch is outside `[0, n_note)`.
6. For each selected note, calculate duration as `offset_idx - onset_idx`, convert it to an integer, then clamp it to `[1, n_duration]`. Thus zero/negative durations become 1, and durations longer than the vocabulary maximum are truncated to that maximum.
7. Convert onset to segment-relative time: `time = onset_idx - start`.
8. Deduplicate by `(relative_onset, pitch)`. If multiple selected notes have the same onset and pitch, retain the one with the longest *clamped* duration. The velocity retained is the velocity associated with that longest duration. If durations tie, the first encountered note remains.
9. Sort retained notes by `(relative_onset, pitch)`, ascending.
10. Emit tokens in that order:
    - Emit `TIME(time)` when the onset differs from the previous note's onset. It is emitted once for all notes at that onset.
    - If `velocity_bins > 0`, map the note's MIDI velocity to a velocity bin. Emit `VELOCITY(bin)` only if that bin differs from the previous emitted bin. The previous bin is tracked across the whole segment, not reset at each onset.
    - Emit `PITCH(pitch)`.
    - Emit `DURATION(duration - 1)`.
11. Append `EOS` if `add_eos=True`.
12. Return a NumPy integer array of token IDs.

### Velocity-bin calculation

If `n_velocity > 0`, the encoder computes:

```python
bin = clip((int(velocity) - 1) * n_velocity // 127, 0, n_velocity - 1)
```

The result is clamped to a valid bin index.

The bin-to-velocity conversion used by the decoder is:

```python
velocity = clip(round((bin + 0.5) * 127 / n_velocity), 1, 127)
```

This is a quantized representation: decoding generally returns a representative velocity for the bin, not necessarily the original velocity.

If velocity binning is disabled (`velocity_bins=0`), no velocity tokens are emitted and decoding uses `DEFAULT_VELOCITY = 77`.

### Encoding example

Suppose a segment starts at beat step `8`, and the selected notes are:

```text
(onset=8,  offset=12, pitch=60, velocity=90)
(onset=8,  offset=10, pitch=64, velocity=90)
(onset=11, offset=14, pitch=67, velocity=70)
```

With velocity binning disabled, the logical token sequence is:

```text
TIME(0) PITCH(60) DURATION(3)
        PITCH(64) DURATION(1)
TIME(3) PITCH(67) DURATION(2)
EOS
```

Durations are displayed here as actual durations (4, 2, and 3 steps respectively); the stored duration token values are one less (3, 1, and 2). The time values are relative to the segment start. Notes at the same onset share a `TIME` token and are sorted by pitch.

## 5. Decoding a segment: `tokens_to_notes`

Signature:

```python
tokens_to_notes(tokens, start_idx=0, n_steps=None)
```

Returns rows of the form:

```text
(absolute_onset, absolute_offset, pitch, velocity)
```

### State maintained by the decoder

- `time`: current segment-relative onset position, initially `0`.
- `velocity`: current decoded velocity, initially `77`.
- `pitch`: pending pitch awaiting a duration, initially `None`.
- `found`: map keyed by `(time, pitch)` storing the selected duration and velocity.

### Step-by-step behavior

1. Read tokens in order and detokenize each ID into `(token_type, value)`.
2. On `EOS`, stop decoding.
3. On `TIME(value)`, set the current time to `value` (this is an absolute segment-relative time, not a shift) and clear any pending pitch.
4. On `VELOCITY(bin)`, convert the bin to its representative MIDI velocity and update the current velocity.
5. On `PITCH(p)`, set the pending pitch to `p`.
6. On `DURATION(value)`, if there is no pending pitch, ignore the duration. If `n_steps` is set and the current time is greater than or equal to `n_steps`, discard the note. Otherwise, the duration is `value + 1`, and the note is stored under `(time, pitch)`. If the same `(time, pitch)` occurs more than once, retain the longest duration. Then clear the pending pitch.
7. A pitch token that is not followed by a duration before another `TIME` token is cleared and does not produce a note. Other malformed token sequences are handled without raising for the cases covered by the implementation.
8. Convert each stored note to absolute indices:
   - `absolute_onset = start_idx + time`
   - `absolute_offset = start_idx + time + duration`
9. Return notes sorted by `(time, pitch)` as a NumPy integer array with shape `(N, 4)`. If no notes are decoded, return an empty array shaped `(0, 4)`.

The decoder does not clip the note offset to `n_steps`; `n_steps` is used to drop notes whose onset lies outside the segment. This matches the encoding rule that a note belongs to a segment based on onset, and its duration can extend beyond the segment.

If no velocity token has been seen, the note uses velocity 77. If velocity tokens are present, their decoded velocity applies to subsequent notes until another velocity token changes it.

## 6. Converting a token sequence to MIDI

The implementation also provides:

```python
relative_batch_tokens_to_midi(
    tokens,
    beatstep,
    beat_offset_idx=None,
    bars_per_batch=None,
    cutoff_time_idx=None,
    max_onset_idx=None,
)
```

This combines decoded segments into one `pretty_midi.PrettyMIDI` object and returns `(midi, notes)`.

Behavior:

1. `beat_offset_idx` defaults to `0`.
2. `bars_per_batch` defaults to `2`.
3. Each segment is assumed to contain `bars_per_batch * 4` beat steps. The default is therefore 8 steps per segment.
4. Segment `n` is decoded with `start_idx = beat_offset_idx + n * n_steps`, where `n_steps = bars_per_batch * 4`.
5. `n_steps` is passed to `tokens_to_notes`, so notes whose onset lies at or beyond the segment's local length are dropped.
6. The decoded note arrays are concatenated.
7. If `max_onset_idx` is set, notes whose absolute onset is greater than or equal to it are removed.
8. Notes are converted to MIDI times using `beatstep`.
9. The returned note rows use absolute beat-step indices; the MIDI object's note start/end times are seconds.

`cutoff_time_idx` is accepted for compatibility but is unused in this implementation.

### MIDI conversion details

`notes_to_midi(notes, beatstep, offset_sec=None)` creates a `pretty_midi.PrettyMIDI` object with resolution 384 and initial tempo 120 BPM. It creates one instrument with program 0.

For each note `(onset_idx, offset_idx, pitch, velocity)`:

- Both indices are clamped to the last valid index of `beatstep`.
- Start time is `beatstep[onset_idx] - offset_sec`.
- End time is `beatstep[offset_idx] - offset_sec`.
- `offset_sec` defaults to `0.0`.

The method then calls `remove_invalid_notes()` and returns the MIDI object.

## 7. Token inspection helpers

### `tokenize_note(idx, token_type)`

Converts a token value to its integer token ID by adding the token type's base ID. It validates the token type and value range.

### `detokenize(token)`

Converts an integer token ID to `[token_type, value]`. Token IDs outside the configured vocabulary are returned as `[TOKEN_SPECIAL, -1]`.

### `to_string(tokens)`

Returns readable `(type, value)` pairs for inspection. Duration values are displayed as actual durations (`stored_value + 1`), while special values are displayed as `PAD`, `EOS`, `TIE`, or `Unknown Special`.

## 8. Important constraints and edge cases

- **Time values:** The time vocabulary must be large enough for the segment length. The encoder rejects segments longer than `n_time`.
- **Duration values:** Durations are clamped to 1 through `n_duration`; longer notes lose duration information.
- **Pitch range:** Pitches outside the configured note vocabulary are skipped during encoding.
- **Duplicate onset/pitch pairs:** Only the longest clamped duration is retained.
- **Velocity:** With no velocity bins, all decoded notes default to velocity 77. With bins enabled, velocity is quantized.
- **Segment ownership:** A note belongs to the segment containing its onset, even if it ends in a later segment.
- **No note-off tokens:** Note endings are encoded only in duration tokens.
- **EOS:** The encoder appends `EOS` by default; the decoder stops at the first `EOS`.
- **Malformed pairs:** A duration without a pending pitch is ignored. A pitch cleared by a new time token before its duration is ignored.
- **Beatstep length:** MIDI conversion clamps indices to the last available beatstep, so insufficient beatstep data can shorten or collapse note times.
- **Configuration mismatch:** The model vocabulary must be at least `tokenizer.vocab_total`, and model/tokenizer token layouts must match.

## 9. Summary

The tokenizer assigns each note to a segment by onset. Within that segment, it represents the note with an absolute segment-relative `TIME`, a `PITCH`, and a zero-based `DURATION` token. Optional velocity-bin tokens describe dynamics. Notes sharing an onset share a time token and are sorted by pitch. The duration is never clipped to the segment boundary, but it is capped by the configured duration vocabulary. Decoding reconstructs absolute onset/offset beat-step indices, and those indices can then be converted into a MIDI object.
