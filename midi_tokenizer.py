"""Onset + duration MIDI tokenizer for beat-aligned segments.

A segment of ``S`` beat steps *owns* the notes whose onset lies inside it. Its
target is the sequence

    TIME(t) [VELOCITY(v)] PITCH(p) DURATION(d)  PITCH(p) DURATION(d) ...
    TIME(t') ...                                                          EOS

* ``TIME(t)``: absolute beat-step position of the onset inside the segment,
  emitted once per distinct onset, in increasing order.
* ``PITCH(p)`` is always followed by ``DURATION(d)``: the note's length in beat
  steps (1..``duration`` vocabulary, longer notes are capped). Notes that share
  an onset are sorted by pitch.
* ``VELOCITY(v)`` (optional, ``velocity_bins > 0``) is emitted only when the
  velocity bin changes.

Compared with the previous on/off tokenizer:

* no note-off events, velocity toggles or offset time tokens (about 25-35 %
  fewer tokens);
* durations are never clipped at the segment border, and notes held from the
  previous segment are not repeated, so there are no cut notes or re-attacks
  between neighbouring segments and no overlap/stitching is needed;
* a canonical event order (onset, then pitch);
* a note is always a complete (pitch, duration) pair, so a damaged sequence loses
  single notes instead of leaving hanging notes.

Vocabulary layout (ids): ``[special][pitch][time][duration][velocity]``. The T5
``vocab_size`` must be at least ``tokenizer.vocab_total``.

Config (``config.tokenizer.vocab_size``): ``special`` (4), ``note`` (128),
``time`` (existing key, must be >= the segment length in beat steps),
``duration`` (default 32) and ``velocity_bins`` (default 0 = no dynamics). The
old ``velocity`` key is ignored.
"""

import numpy as np
import pretty_midi
import scipy.interpolate as interp

TOKEN_SPECIAL: int = 0
TOKEN_NOTE: int = 1
TOKEN_VELOCITY: int = 2
TOKEN_TIME: int = 3
TOKEN_DURATION: int = 4

DEFAULT_VELOCITY: int = 77

TIE: int = 2  # unused, kept so the special-token layout is unchanged
EOS: int = 1
PAD: int = 0


def extrapolate_beat_times(beat_times, n_extend=1):
    beat_times_function = interp.interp1d(
        np.arange(beat_times.size),
        beat_times,
        bounds_error=False,
        fill_value="extrapolate",
    )

    ext_beats = beat_times_function(
        np.linspace(0, beat_times.size + n_extend - 1, beat_times.size + n_extend)
    )

    return ext_beats


class MidiTokenizer:
    def __init__(self, config) -> None:
        """``config`` is the ``tokenizer`` section of the config (with ``vocab_size``)."""
        self.config = config
        sizes = config.vocab_size
        self.n_special = int(sizes.get("special", 4))
        self.n_note = int(sizes.get("note", 128))
        self.n_time = int(sizes.get("time", 100))
        self.n_duration = int(sizes.get("duration", 32))
        self.n_velocity = int(sizes.get("velocity_bins", 0))

        self._size = {
            TOKEN_SPECIAL: self.n_special,
            TOKEN_NOTE: self.n_note,
            TOKEN_TIME: self.n_time,
            TOKEN_DURATION: self.n_duration,
            TOKEN_VELOCITY: self.n_velocity,
        }
        self._base = {}
        position = 0
        for token_type in (
            TOKEN_SPECIAL,
            TOKEN_NOTE,
            TOKEN_TIME,
            TOKEN_DURATION,
            TOKEN_VELOCITY,
        ):
            self._base[token_type] = position
            position += self._size[token_type]
        self.vocab_total = position

    # ------------------------------------------------------------------ #
    # single tokens
    # ------------------------------------------------------------------ #
    def tokenize_note(self, idx, token_type):
        """Encode a value of the given type as one token id."""
        if token_type not in self._size:
            raise ValueError(f"type {token_type} is not a predefined token type.")
        if not 0 <= idx < self._size[token_type]:
            raise ValueError(
                f"value {idx} out of range for token type {token_type} "
                f"(size {self._size[token_type]})"
            )
        return self._base[token_type] + int(idx)

    def detokenize(self, token):
        """Decode one token id into ``[type, value]``.

        Ids outside the vocabulary (a T5 head larger than the vocabulary) are
        returned as an unknown special token ``[TOKEN_SPECIAL, -1]``.
        """
        token = int(token)
        for token_type in (
            TOKEN_VELOCITY,
            TOKEN_DURATION,
            TOKEN_TIME,
            TOKEN_NOTE,
            TOKEN_SPECIAL,
        ):
            size = self._size[token_type]
            base = self._base[token_type]
            if size and base <= token < base + size:
                return [token_type, token - base]
        return [TOKEN_SPECIAL, -1]

    def to_string(self, tokens):
        """Readable ``(type, value)`` pairs, for debugging."""
        names = {
            TOKEN_NOTE: "pitch",
            TOKEN_VELOCITY: "velocity",
            TOKEN_TIME: "time",
            TOKEN_DURATION: "duration",
        }
        out = []
        for token in tokens:
            token_type, value = self.detokenize(token)
            if token_type == TOKEN_SPECIAL:
                value = {EOS: "EOS", PAD: "PAD", TIE: "TIE"}.get(value, "Unknown Special")
                out.append(("special", value))
            elif token_type == TOKEN_DURATION:
                out.append(("duration", value + 1))
            else:
                out.append((names[token_type], value))
        return out

    # ------------------------------------------------------------------ #
    # velocity bins
    # ------------------------------------------------------------------ #
    def _velocity_to_bin(self, velocity):
        return int(np.clip((int(velocity) - 1) * self.n_velocity // 127, 0, self.n_velocity - 1))

    def _bin_to_velocity(self, velocity_bin):
        return int(np.clip(round((velocity_bin + 0.5) * 127 / self.n_velocity), 1, 127))

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def segment_to_tokens(self, notes, start, end, add_eos=True):
        """Target tokens of the segment ``[start, end)`` (absolute beat steps).

        Args:
            notes: ``(onset_idx, offset_idx, pitch, velocity)`` rows for the
                whole song (not clipped).
            start, end: segment bounds in beat steps. Only notes whose onset is
                in ``[start, end)`` are encoded; their durations are *not*
                clipped to the segment.
            add_eos: append the end-of-sequence token.
        """
        start, end = int(start), int(end)
        if end - start > self.n_time:
            raise ValueError(
                f"segment of {end - start} steps does not fit the time "
                f"vocabulary ({self.n_time})"
            )
        notes = np.asarray(notes)
        best = {}  # (onset, pitch) -> (duration, velocity); keep the longest
        if notes.ndim == 2 and len(notes):
            selected = notes[(notes[:, 0] >= start) & (notes[:, 0] < end)]
            for onset, offset, pitch, velocity in selected[:, :4]:
                pitch = int(pitch)
                if not 0 <= pitch < self.n_note:
                    continue
                duration = int(np.clip(int(offset) - int(onset), 1, self.n_duration))
                key = (int(onset) - start, pitch)
                if key not in best or duration > best[key][0]:
                    best[key] = (duration, int(velocity))

        tokens = []
        previous_time = None
        previous_bin = None
        for (time, pitch), (duration, velocity) in sorted(best.items()):
            if time != previous_time:
                tokens.append(self.tokenize_note(time, TOKEN_TIME))
                previous_time = time
            if self.n_velocity:
                velocity_bin = self._velocity_to_bin(velocity)
                if velocity_bin != previous_bin:
                    tokens.append(self.tokenize_note(velocity_bin, TOKEN_VELOCITY))
                    previous_bin = velocity_bin
            tokens.append(self.tokenize_note(pitch, TOKEN_NOTE))
            tokens.append(self.tokenize_note(duration - 1, TOKEN_DURATION))
        if add_eos:
            tokens.append(EOS)
        return np.array(tokens, dtype=int)

    # ------------------------------------------------------------------ #
    # decoding
    # ------------------------------------------------------------------ #
    def tokens_to_notes(self, tokens, start_idx=0, n_steps=None):
        """Decode one segment's tokens into ``(onset, offset, pitch, velocity)`` rows.

        Indices are absolute (``start_idx`` + position in the segment). Notes
        whose onset position is >= ``n_steps`` belong to the next segment and are
        dropped. Malformed sequences never raise: a pitch without a duration is
        ignored, and repeated (onset, pitch) notes keep the longest duration.
        """
        start_idx = int(start_idx)
        time = 0
        velocity = DEFAULT_VELOCITY
        pitch = None
        found = {}
        for token in tokens:
            token_type, value = self.detokenize(token)
            if token_type == TOKEN_SPECIAL:
                if value == EOS:
                    break
            elif token_type == TOKEN_TIME:
                time = value
                pitch = None
            elif token_type == TOKEN_VELOCITY:
                velocity = self._bin_to_velocity(value)
            elif token_type == TOKEN_NOTE:
                pitch = value
            elif token_type == TOKEN_DURATION:
                if pitch is None or (n_steps is not None and time >= n_steps):
                    pitch = None
                    continue
                key = (time, pitch)
                duration = value + 1
                if key not in found or duration > found[key][0]:
                    found[key] = (duration, velocity)
                pitch = None

        rows = [
            [start_idx + time, start_idx + time + duration, pitch, velocity]
            for (time, pitch), (duration, velocity) in sorted(found.items())
        ]
        return np.array(rows, dtype=int).reshape(-1, 4)

    def relative_batch_tokens_to_midi(
        self,
        tokens,
        beatstep,
        beat_offset_idx=None,
        bars_per_batch=None,
        cutoff_time_idx=None,
        max_onset_idx=None,
    ):
        """Decode consecutive segments into one MIDI object.

        Args:
            tokens: one token sequence per segment; segment ``n`` starts at beat
                step ``beat_offset_idx + n * bars_per_batch * 4``.
            beatstep: beat times (seconds), long enough to cover the notes.
            bars_per_batch: segment length in bars (4 steps each).
            cutoff_time_idx: unused, kept for call compatibility.
            max_onset_idx: drop notes starting at or after this beat step (for
                example the end of the audio).

        Returns ``(pretty_midi.PrettyMIDI, notes)``.
        """
        beat_offset_idx = 0 if beat_offset_idx is None else beat_offset_idx
        bars_per_batch = 2 if bars_per_batch is None else bars_per_batch
        n_steps = bars_per_batch * 4

        parts = []
        for n in range(len(tokens)):
            notes = self.tokens_to_notes(
                tokens[n], start_idx=beat_offset_idx + n * n_steps, n_steps=n_steps
            )
            if len(notes):
                parts.append(notes)
        notes = np.concatenate(parts) if parts else np.zeros((0, 4), dtype=int)
        if max_onset_idx is not None:
            notes = notes[notes[:, 0] < max_onset_idx]
        midi = self.notes_to_midi(notes, beatstep, offset_sec=beatstep[beat_offset_idx])
        return midi, notes

    def notes_to_midi(self, notes, beatstep, offset_sec=None):
        """Create a piano MIDI object from indexed note events and beat times.

        Indices past the end of ``beatstep`` are clamped to its last entry.
        """
        new_pm = pretty_midi.PrettyMIDI(resolution=384, initial_tempo=120.0)
        new_inst = pretty_midi.Instrument(program=0)
        if offset_sec is None:
            offset_sec = 0.0
        last = len(beatstep) - 1

        new_notes = []
        for onset_idx, offset_idx, pitch, velocity in notes:
            onset_idx = min(int(onset_idx), last)
            offset_idx = min(int(offset_idx), last)
            new_notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=float(beatstep[onset_idx] - offset_sec),
                    end=float(beatstep[offset_idx] - offset_sec),
                )
            )
        new_inst.notes = new_notes
        new_pm.instruments.append(new_inst)
        new_pm.remove_invalid_notes()
        return new_pm