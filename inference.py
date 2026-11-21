"""Turn pop songs into piano covers.

    python inference.py JIQiGN-vO-g.pitchshift.wav --checkpoint checkpoints/lightning_logs/version_2/checkpoints/last.ckpt
python inference.py 8Sc4Pb7d1Nk.pitchshift.wav --checkpoint "checkpoints/lightning_logs/version_4/checkpoints/epoch=9-val_loss=1.3755.ckpt" --save_mix
writes ``output/song.mid``.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from transformer_wrapper import TransformerWrapper


def load_model(config, checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    model = TransformerWrapper(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def main(args):
    config = OmegaConf.load(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(config, args.checkpoint, device)

    n_bars = args.n_bars or config.dataset.n_bars
    if n_bars <= 0:
        raise ValueError("n_bars must be positive")
    if args.max_batch_size is not None and args.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if args.num_beams <= 0:
        raise ValueError("num_beams must be positive")
    if n_bars * 4 > model.tokenizer.n_time:
        raise ValueError(
            f"n_bars={n_bars} needs {n_bars * 4} time tokens, "
            f"but the tokenizer provides {model.tokenizer.n_time}"
        )

    if args.greedy:
        generation_kwargs = {"do_sample": False, "num_beams": 1}
    else:
        generation_kwargs = {
            "do_sample": args.num_beams == 1,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio in args.audio:
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        midi_path = output_dir / f"{audio_path.stem}.mid"
        mix_path = output_dir / f"{audio_path.stem}.mix.wav"
        print(f"{audio_path} -> {midi_path}")

        model.generate(
            audio_path=str(audio_path),
            composer=args.composer,
            n_bars=n_bars,
            max_batch_size=args.max_batch_size,
            max_length=args.max_length,
            generation_kwargs=generation_kwargs,
            seed=args.seed,
            save_midi=True,
            midi_path=str(midi_path),
            save_mix=args.save_mix,
            mix_path=str(mix_path) if args.save_mix else None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a piano cover (MIDI) from audio files."
    )
    parser.add_argument("audio", nargs="+", help="path/to/audio_name.ext")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--composer", default="composer1")
    parser.add_argument(
        "--n_bars", type=int, default=None, help="default: dataset.n_bars"
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="chunks processed at once (default: 64 // n_bars)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="maximum generated target length (default: config.dataset.target_length)",
    )
    parser.add_argument("--greedy", action="store_true", help="disable sampling")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save_mix",
        action="store_true",
        help="also write output/<name>.mix.wav (song + rendered piano, stereo)",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    main(parser.parse_args())