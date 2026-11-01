"""Pop2Piano-style pop audio -> piano MIDI model with a pretrained audio encoder.

Differences from the original wrapper
-------------------------------------
* The log-mel front end is replaced by a pretrained music encoder (MERT by
  default). Its per-layer hidden states are mixed with learned weights, then
  projected to the T5 ``d_model``.
* Each chunk is encoded together with ``context_seconds`` of neighbouring audio;
  only the frames that fall inside the chunk are kept, so the encoder sees
  musical context across chunk boundaries.
* A beat-position embedding (position of every frame on the chunk's beat grid)
  is added to the encoder inputs. This is the same grid the output tokens use.
* The composer embedding is prepended as one extra encoder frame.
* An attention mask is built from the real chunk lengths and passed to T5
  (training and generation), so padding is never attended to.
* Decoding is configurable (sampling by default) and chunks can be generated
  with an overlap: only the notes in the middle of each chunk are kept.
* Bug fixes: missing f-string, ``y`` undefined when ``beatsteps`` is given,
  dead ``feature_tokens`` path removed.
"""

from __future__ import annotations

import contextlib
import math
import os
import random

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, T5Config, T5ForConditionalGeneration

from midi_tokenizer import MidiTokenizer, extrapolate_beat_times
from utils.dsp import get_stereo


DEFAULT_COMPOSERS = {"various composer": 2052}

# Used when the config has no ``encoder:`` block (or only part of it).
DEFAULT_ENCODER_CONFIG = {
    "name": "m-a-p/MERT-v1-95M",
    "revision": None,  # pin a commit hash here: trust_remote_code runs hub code
    "sample_rate": 24000,
    "freeze": True,
    "normalize_waveform": True,
    "context_seconds": 2.0,  # audio taken on each side of a chunk
    "lr_mult": 0.1,  # LR multiplier for the encoder when freeze is False
}

# Periods (in beat steps) of the sin/cos features describing a frame's position
# on the chunk's beat grid. Period 1 = position inside a step.
BEAT_PERIODS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
N_BEAT_FEATURES = 2 * len(BEAT_PERIODS) + 1  # sin/cos per period + log step duration

DEFAULT_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 0.95,
    "top_k": 0,
    "top_p": 0.95,
    "num_beams": 1,
}


def get_encoder_config(config):
    """Encoder settings from ``config.encoder`` merged over the defaults."""
    base = OmegaConf.create(DEFAULT_ENCODER_CONFIG)
    user = config.get("encoder")
    return OmegaConf.merge(base, user) if user is not None else base


def _fit_grid(grid: torch.Tensor, length: int) -> torch.Tensor:
    """Trim or extend (repeating the last interval) a beat grid to ``length``."""
    if grid.shape[0] >= length:
        return grid[:length]
    step = (grid[-1] - grid[-2]).clamp_min(1e-3)
    extra = grid[-1] + step * torch.arange(
        1, length - grid.shape[0] + 1, device=grid.device, dtype=grid.dtype
    )
    return torch.cat([grid, extra])


class PretrainedAudioEncoder(nn.Module):
    """MERT (or any wav2vec2-style model) -> mixed hidden states per frame.

    The frame rate is sample_rate / 320 (75 Hz for MERT at 24 kHz). Waveforms in
    a batch are zero-padded and *no* attention mask is given to the model, which
    is what group-norm wav2vec2-style models expect; the padded frames are
    masked out later by our own mask.
    """

    HOP = 320
    RECEPTIVE_FIELD = 400
    MIN_SAMPLES = 2400

    def __init__(self, cfg):
        super().__init__()
        self.sample_rate = int(cfg.sample_rate)
        self.freeze = bool(cfg.freeze)
        self.normalize = bool(cfg.normalize_waveform)

        self.model = AutoModel.from_pretrained(
            cfg.name, trust_remote_code=True, revision=cfg.revision
        )
        self.hidden_size = int(self.model.config.hidden_size)
        self.n_layers = int(self.model.config.num_hidden_layers) + 1
        # Learned softmax mixing weights over the embedding + transformer layers.
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def forward(self, wave: torch.Tensor, lengths: torch.Tensor):
        """
        wave    : (B, L) zero-padded waveforms at ``self.sample_rate``
        lengths : (B,) real number of samples of each waveform

        Returns (features (B, T, D), frame_times (T,) in seconds from the start
        of each window, valid (B, T) bool).
        """
        if wave.shape[1] < self.MIN_SAMPLES:
            wave = torch.nn.functional.pad(wave, (0, self.MIN_SAMPLES - wave.shape[1]))
        length = wave.shape[1]
        mask = (
            torch.arange(length, device=wave.device)[None] < lengths[:, None]
        ).to(wave.dtype)

        if self.normalize:
            n = lengths.clamp_min(1).to(wave.dtype)[:, None]
            mean = (wave * mask).sum(1, keepdim=True) / n
            var = (((wave - mean) * mask) ** 2).sum(1, keepdim=True) / n
            wave = (wave - mean) / torch.sqrt(var + 1e-7)
        wave = wave * mask

        context = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with context:
            hidden = self.model(wave, output_hidden_states=True).hidden_states
        assert len(hidden) == self.n_layers, (len(hidden), self.n_layers)

        weights = torch.softmax(self.layer_logits, dim=0)
        features = sum(w * h for w, h in zip(weights, hidden))

        n_frames = features.shape[1]
        frame_times = (
            torch.arange(n_frames, device=wave.device, dtype=torch.float32) * self.HOP
            + self.RECEPTIVE_FIELD / 2
        ) / self.sample_rate
        valid = frame_times[None] < (lengths.to(torch.float32) / self.sample_rate)[:, None]
        return features, frame_times, valid


class TransformerWrapper(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tokenizer = MidiTokenizer(config.tokenizer)
        self.t5config = T5Config.from_pretrained("t5-small")
        for k, v in config.t5.items():
            setattr(self.t5config, k, v)
        self.transformer = T5ForConditionalGeneration(self.t5config)
        d_model = self.t5config.d_model

        # Audio front end.
        self.encoder_cfg = get_encoder_config(config)
        self.sample_rate = int(self.encoder_cfg.sample_rate)
        self.context_seconds = float(self.encoder_cfg.context_seconds)
        self.audio_encoder = PretrainedAudioEncoder(self.encoder_cfg)
        self.feat_norm = nn.LayerNorm(self.audio_encoder.hidden_size)
        self.audio_proj = nn.Linear(self.audio_encoder.hidden_size, d_model)
        self.beat_proj = nn.Linear(N_BEAT_FEATURES, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "beat_periods", torch.tensor(BEAT_PERIODS), persistent=False
        )

        # Composer conditioning (one extra encoder frame).
        self.composer_conditioned = bool(
            config.dataset.get("mel_is_conditioned", True)
        )
        if self.composer_conditioned:
            self.composer_to_feature_token = dict(config.composer_to_feature_token)
            values = list(self.composer_to_feature_token.values())
            self.composer_offset = min(values)
            self.composer_embedding = nn.Embedding(
                max(values) - min(values) + 1, d_model
            )
        else:
            self.composer_to_feature_token = dict(DEFAULT_COMPOSERS)

        self.lr = config.training.lr

    # ------------------------------------------------------------------ #
    # Training-mode handling
    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.audio_encoder.freeze:
            self.audio_encoder.model.eval()  # no dropout / spec-augment when frozen
        return self

    # ------------------------------------------------------------------ #
    # Encoder inputs
    # ------------------------------------------------------------------ #
    def beat_features(self, tau: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Position of frames on a chunk's beat grid.

        tau  : (N,) frame times in seconds from the chunk start
        grid : (S+1,) beat-step times in seconds from the chunk start (grid[0]=0)

        Returns (N, N_BEAT_FEATURES): sin/cos of the position (in steps) at
        several periods, plus log of the local step duration (tempo cue).
        """
        n_steps = grid.shape[0] - 1
        grid = grid.contiguous().to(tau.dtype)
        idx = (torch.searchsorted(grid, tau.contiguous(), right=True) - 1).clamp(
            0, n_steps - 1
        )
        left, right = grid[idx], grid[idx + 1]
        step_dur = (right - left).clamp_min(1e-3)
        phase = ((tau - left) / step_dur).clamp(0.0, 1.0)
        position = idx.to(tau.dtype) + phase
        angles = 2.0 * math.pi * position[:, None] / self.beat_periods[None].to(tau.dtype)
        return torch.cat(
            [angles.sin(), angles.cos(), step_dur.log()[:, None]], dim=-1
        )

    def build_encoder_inputs(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        offsets: torch.Tensor,
        durations: torch.Tensor,
        beat_grids: torch.Tensor,
        composer: torch.Tensor | None = None,
    ):
        """Turn audio windows into T5 ``inputs_embeds`` and ``attention_mask``.

        audio         : (B, L) zero-padded windows (chunk + context) at encoder SR
        audio_lengths : (B,) real lengths in samples
        offsets       : (B,) chunk start inside its window, seconds
        durations     : (B,) chunk length, seconds
        beat_grids    : (B, S+1) beat-step times relative to the chunk start
        composer      : (B,) composer token values (only if conditioned)
        """
        features, frame_times, valid = self.audio_encoder(audio, audio_lengths)
        features = self.audio_proj(self.feat_norm(features))

        offsets = offsets.to(frame_times.device, frame_times.dtype)
        durations = durations.to(frame_times.device, frame_times.dtype)

        sequences = []
        for i in range(features.shape[0]):
            lo = offsets[i]
            keep = valid[i] & (frame_times >= lo) & (frame_times < lo + durations[i])
            idx = keep.nonzero(as_tuple=True)[0]
            tau = frame_times[idx] - lo
            beat = self.beat_proj(self.beat_features(tau, beat_grids[i]))
            sequences.append(self.input_norm(features[i, idx] + beat))

        if self.composer_conditioned:
            if composer is None:
                raise ValueError("composer is required when conditioning is on")
            composer_frames = self.composer_embedding(
                composer.to(features.device) - self.composer_offset
            )
            sequences = [
                torch.cat([composer_frames[i : i + 1].to(s.dtype), s])
                for i, s in enumerate(sequences)
            ]

        lengths = torch.tensor([s.shape[0] for s in sequences], device=features.device)
        inputs_embeds = pad_sequence(sequences, batch_first=True)
        attention_mask = (
            torch.arange(inputs_embeds.shape[1], device=features.device)[None]
            < lengths[:, None]
        ).long()
        return inputs_embeds, attention_mask

    def encode_batch(self, batch: dict):
        """Encoder inputs from a collated batch (see dataset.collate_batch)."""
        return self.build_encoder_inputs(
            batch["audio"],
            batch["audio_lengths"],
            batch["offsets"],
            batch["durations"],
            batch["beat_grids"],
            batch.get("composer"),
        )

    def forward(self, batch: dict):
        inputs_embeds, attention_mask = self.encode_batch(batch)
        return self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=batch.get("labels"),
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _make_chunks(self, audio, ext_beatstep, starts, n_steps, n_target_step):
        """Audio windows (chunk + context) and beat grids for each chunk start."""
        sr = self.sample_rate
        ctx = self.context_seconds
        ext_np = np.asarray(ext_beatstep, dtype=np.float64)
        audio_dur = audio.shape[0] / sr

        windows, offsets, durations, grids = [], [], [], []
        for s in starts:
            e = min(s + n_steps, n_target_step)
            t_s, t_e = ext_np[s], ext_np[e]
            win_start = max(0.0, t_s - ctx)
            win_end = min(audio_dur, t_e + ctx)
            a0 = min(int(win_start * sr), audio.shape[0] - 1)
            a1 = max(a0 + 1, min(int(win_end * sr), audio.shape[0]))
            windows.append(audio[a0:a1])
            offsets.append(t_s - a0 / sr)
            durations.append(t_e - t_s)
            grid = torch.as_tensor(
                ext_np[s : s + n_steps + 1] - t_s,
                dtype=torch.float32,
                device=audio.device,
            )
            grids.append(_fit_grid(grid, n_steps + 1))
        return windows, offsets, durations, grids

    def _decode_with_overlap(
        self, relative_tokens, ext_beatstep, starts, n_bars, n_steps, overlap_steps
    ):
        """Decode chunks one by one and keep each chunk's middle notes only.

        Chunk ``i`` starts at beat step ``starts[i]`` and "owns" the notes whose
        onset lies in [s_i + margin, s_i + margin + stride) (the first chunk
        also owns everything before, the last one everything after). Notes keep
        their full length even if they ring into the next chunk's region.
        """
        margin = overlap_steps // 2
        stride = n_steps - overlap_steps
        merged, pm_ref = [], None
        for i, s in enumerate(starts):
            pm_i, _ = self.tokenizer.relative_batch_tokens_to_midi(
                relative_tokens[i : i + 1],
                beatstep=ext_beatstep[s:],
                bars_per_batch=n_bars,
                cutoff_time_idx=(n_bars + 1) * 4,
            )
            if pm_ref is None:
                pm_ref = pm_i
            lo = 0.0 if i == 0 else float(ext_beatstep[s + margin])
            hi = (
                float("inf")
                if i == len(starts) - 1
                else float(ext_beatstep[s + margin + stride])
            )
            merged.extend(n for n in pm_i.instruments[0].notes if lo <= n.start < hi)
        merged.sort(key=lambda n: (n.start, n.pitch))
        pm_ref.instruments[0].notes = merged
        return pm_ref

    @torch.no_grad()
    def single_inference(
        self,
        audio,
        beatstep,
        max_length=256,
        max_batch_size=64,
        n_bars=None,
        composer_value=None,
        overlap_steps=0,
        generation_kwargs=None,
    ):
        """Generate MIDI for a whole song.

        audio    : 1-D float tensor at ``self.sample_rate``, already trimmed to
                   [beatstep[0], beatstep[-1]]
        beatstep : (time,) beat-step times in seconds, beatstep[0] == 0
        overlap_steps : beat steps shared by consecutive chunks (0 = disjoint
                   chunks decoded in one tokenizer call, like the original)
        """
        assert audio.ndim == 1
        n_bars = self.config.dataset.n_bars if n_bars is None else n_bars
        n_steps = n_bars * 4
        overlap_steps = int(overlap_steps)
        if not 0 <= overlap_steps < n_steps:
            raise ValueError(f"overlap_steps must be in [0, {n_steps})")
        PAD = self.t5config.pad_token_id

        beatstep = np.asarray(beatstep, dtype=np.float64)
        if beatstep[0] > 0.01:
            print(
                f"inference warning : beatstep[0] is not 0 ({beatstep[0]}). "
                "all beatstep will be shifted."
            )
            beatstep = beatstep - beatstep[0]

        ext_beatstep = extrapolate_beat_times(beatstep, (n_bars + 1) * 4 + 1)
        n_target_step = len(beatstep)
        stride = n_steps - overlap_steps
        margin = overlap_steps // 2
        starts = [
            s
            for s in range(0, n_target_step, stride)
            if s == 0 or s + margin < n_target_step
        ]

        windows, offsets, durations, grids = self._make_chunks(
            audio, ext_beatstep, starts, n_steps, n_target_step
        )

        gen_kwargs = dict(DEFAULT_GENERATION_KWARGS)
        gen_kwargs.update(generation_kwargs or {})
        if not gen_kwargs.get("do_sample", False):
            for key in ("temperature", "top_k", "top_p"):
                gen_kwargs.pop(key, None)

        relative_tokens = []
        for b0 in range(0, len(starts), max_batch_size):
            sl = slice(b0, b0 + max_batch_size)
            chunk_windows = windows[sl]
            wave = pad_sequence(chunk_windows, batch_first=True)
            lengths = torch.tensor(
                [w.shape[0] for w in chunk_windows], device=wave.device
            )
            composer = None
            if self.composer_conditioned:
                composer = torch.full(
                    (len(chunk_windows),),
                    int(composer_value),
                    dtype=torch.long,
                    device=wave.device,
                )
            inputs_embeds, attention_mask = self.build_encoder_inputs(
                wave,
                lengths,
                torch.tensor(offsets[sl], dtype=torch.float32, device=wave.device),
                torch.tensor(durations[sl], dtype=torch.float32, device=wave.device),
                torch.stack(grids[sl]),
                composer,
            )
            out = self.transformer.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=max_length,
                **gen_kwargs,
            )
            relative_tokens.append(out.cpu().numpy())

        width = max(rt.shape[-1] for rt in relative_tokens)
        relative_tokens = np.concatenate(
            [
                np.pad(
                    rt, [(0, 0), (0, width - rt.shape[-1])], constant_values=PAD
                )
                for rt in relative_tokens
            ]
        )

        if overlap_steps == 0:
            pm, notes = self.tokenizer.relative_batch_tokens_to_midi(
                relative_tokens,
                beatstep=ext_beatstep,
                bars_per_batch=n_bars,
                cutoff_time_idx=(n_bars + 1) * 4,
            )
        else:
            pm = self._decode_with_overlap(
                relative_tokens, ext_beatstep, starts, n_bars, n_steps, overlap_steps
            )
            notes = None  # the merged result lives in ``pm``
        return relative_tokens, notes, pm

    @torch.no_grad()
    def generate(
        self,
        audio_path=None,
        composer=None,
        model="generated",
        steps_per_beat=2,
        stereo_amp=0.5,
        n_bars=4,
        ignore_duplicate=True,
        show_plot=False,
        save_midi=False,
        save_mix=False,
        midi_path=None,
        mix_path=None,
        click_amp=0.2,
        add_click=False,
        max_batch_size=None,
        beatsteps=None,
        mix_sample_rate=None,
        audio_y=None,
        audio_sr=None,
        overlap_steps=None,
        generation_kwargs=None,
        seed=None,
    ):
        """Cover one song.

        overlap_steps : beat steps shared by neighbouring chunks (default: half
                        a chunk; 0 disables overlap).
        generation_kwargs : overrides for ``T5.generate`` (e.g.
                        ``{"do_sample": False, "num_beams": 4}``).
        seed          : makes sampling reproducible.
        """
        config = self.config

        if audio_path is not None:
            extension = os.path.splitext(audio_path)[1]
            mix_path = (
                audio_path.replace(extension, f".{model}.{composer}.wav")
                if mix_path is None
                else mix_path
            )
            midi_path = (
                audio_path.replace(extension, f".{model}.{composer}.mid")
                if midi_path is None
                else midi_path
            )

        max_batch_size = 64 // n_bars if max_batch_size is None else max_batch_size
        composer_to_feature_token = self.composer_to_feature_token
        if composer is None:
            composer = random.sample(list(composer_to_feature_token.keys()), 1)[0]
        composer_value = composer_to_feature_token[composer]

        if not ignore_duplicate and midi_path is not None and os.path.exists(midi_path):
            return

        if seed is not None:
            torch.manual_seed(seed)

        # --- load audio --------------------------------------------------
        ESSENTIA_SAMPLERATE = 44100
        if audio_y is not None:
            if audio_sr is None:
                raise ValueError("audio_sr is required with audio_y")
            y, sr = audio_y, audio_sr
        elif audio_path is not None:
            y, sr = librosa.load(audio_path, sr=ESSENTIA_SAMPLERATE)
        else:
            raise ValueError("give audio_path or audio_y")

        # --- beats -------------------------------------------------------
        if beatsteps is None:
            from preprocess.beat_quantizer import extract_rhythm, interpolate_beat_times

            y_essentia = (
                y
                if sr == ESSENTIA_SAMPLERATE
                else librosa.resample(y, orig_sr=sr, target_sr=ESSENTIA_SAMPLERATE)
            )
            (
                bpm,
                beat_times,
                confidence,
                estimates,
                essentia_beat_intervals,
            ) = extract_rhythm(audio_path, y=y_essentia)
            beatsteps = interpolate_beat_times(
                np.array(beat_times), steps_per_beat, extend=True
            )
        beatsteps = np.asarray(beatsteps)

        # --- model input at the encoder sample rate ------------------------
        y_model = (
            y
            if sr == self.sample_rate
            else librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        )
        start_sample = int(beatsteps[0] * self.sample_rate)
        end_sample = int(beatsteps[-1] * self.sample_rate)
        _audio = torch.from_numpy(
            np.ascontiguousarray(y_model[start_sample:end_sample], dtype=np.float32)
        ).to(self.device)

        if overlap_steps is None:
            overlap_steps = (n_bars * 4) // 2

        relative_tokens, notes, pm = self.single_inference(
            audio=_audio,
            beatstep=beatsteps - beatsteps[0],
            max_length=config.dataset.target_length
            * max(1, (n_bars // config.dataset.n_bars)),
            max_batch_size=max_batch_size,
            n_bars=n_bars,
            composer_value=composer_value,
            overlap_steps=overlap_steps,
            generation_kwargs=generation_kwargs,
        )

        for n in pm.instruments[0].notes:
            n.start += beatsteps[0]
            n.end += beatsteps[0]

        # --- optional outputs ----------------------------------------------
        if show_plot or save_mix:
            mix_sr = (
                config.dataset.get("sample_rate", self.sample_rate)
                if mix_sample_rate is None
                else mix_sample_rate
            )
            y_mix = (
                y if sr == mix_sr else librosa.resample(y, orig_sr=sr, target_sr=mix_sr)
            )
            if add_click:
                clicks = (
                    librosa.clicks(times=beatsteps, sr=mix_sr, length=len(y_mix))
                    * click_amp
                )
                y_mix = y_mix + clicks
            pm_y = pm.fluidsynth(mix_sr)
            stereo = get_stereo(y_mix, pm_y, pop_scale=stereo_amp)

        if show_plot:
            import IPython.display as ipd
            from IPython.display import display
            import note_seq

            display("Stereo MIX", ipd.Audio(stereo, rate=mix_sr))
            display("Rendered MIDI", ipd.Audio(pm_y, rate=mix_sr))
            display("Original Song", ipd.Audio(y_mix, rate=mix_sr))
            display(note_seq.plot_sequence(note_seq.midi_to_note_sequence(pm)))

        if save_mix:
            sf.write(file=mix_path, data=stereo.T, samplerate=mix_sr, format="wav")

        if save_midi:
            pm.write(midi_path)

        return pm, composer, mix_path, midi_path