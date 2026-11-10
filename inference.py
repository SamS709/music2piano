"""Turn pop songs into piano covers.

    python inference.py JIQiGN-vO-g.pitchshift.wav --checkpoint checkpoints/lightning_logs/version_2/checkpoints/last.ckpt
python inference.py 8Sc4Pb7d1Nk.pitchshift.wav --checkpoint checkpoints/lightning_logs/version_4/checkpoints/epoch\=9-val_loss\=1.3755.ckpt --save_mix
writes ``output/song.mid``.
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from transformer_wrapper import TransformerWrapper


def load_model(config, checkpoint, device):
    model = TransformerWrapper(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    return model.to(device).eval()


def main(args):
    config = OmegaConf.load(args.config)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(config, args.checkpoint, device)

    n_bars = args.n_bars or config.dataset.n_bars
    if args.greedy:
        generation_kwargs = {"do_sample": False}
    else:
        generation_kwargs = {"temperature": args.temperature, "top_p": args.top_p}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio in args.audio:
        audio_path = Path(audio)
        midi_path = output_dir / f"{audio_path.stem}.mid"
        mix_path = output_dir / f"{audio_path.stem}.mix.wav"
        print(f"{audio_path} -> {midi_path}")

        model.generate(
            audio_path=str(audio_path),
            composer=args.composer,
            n_bars=n_bars,
            max_batch_size=args.max_batch_size,
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
        default=8,
        help="chunks processed at once; lower it if you run out of GPU memory",
    )
    parser.add_argument("--greedy", action="store_true", help="disable sampling")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--save_mix",
        action="store_true",
        help="also write output/<name>.mix.wav (song + rendered piano, stereo)",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    main(parser.parse_args())