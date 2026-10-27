"""
Usage:
python youtube_down.py piano_covers.txt /output/dir
"""

import os
import tempfile
import shutil
import pandas as pd
import sys
import subprocess
from pathlib import Path

from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_roots import next_dataset_root


# Run yt-dlp as a module of the *current* Python, so it always uses the
# env that launched the script (no PATH surprises).
YTDLP = [
    sys.executable, "-m", "yt_dlp",
    "--extractor-args", "youtube:player_client=android,ios",
]


def _parse_piano_filename(filename):
    if not filename.endswith(".wav"):
        return None
    parts = filename[:-4].rsplit("___", 3)
    if len(parts) != 4 or not parts[3].isdigit():
        return None
    uploader, title, ytid, duration = parts
    return uploader, title, ytid, int(duration)


def _parse_song_filename(filename):
    if not filename.endswith(".wav"):
        return None
    parts = filename[:-4].rsplit("___", 2)
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    ytid, title, duration = parts
    return ytid, title, int(duration)


def download_piano(
    url: str,
    output_dir: str,
    postprocess=True,
    dry_run=False,
) -> int:
    # os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = f"{tmpdir}/%(uploader)s___%(title)s___%(id)s___%(duration)d.%(ext)s"
        cmd = YTDLP + [
            "-o", output,
            "--extract-audio",
            "--audio-quality", "0",
            "--audio-format", "wav",
            "--retries", "50",
            "--force-ipv4",
            "--yes-playlist",
            "--ignore-errors",
        ]
        if postprocess:
            cmd += ["--postprocessor-args", "ffmpeg:-ac 1 -ar 16000"]
        if dry_run:
            cmd += ["--get-filename"]
        cmd.append(url)
        result = subprocess.call(cmd)

        if not dry_run:
            for filename in os.listdir(tmpdir):
                parsed = _parse_piano_filename(filename)
                if parsed is None:
                    continue
                uploader, title, ytid, duration = parsed
                meta = OmegaConf.create()
                meta.piano = OmegaConf.create()
                meta.piano.uploader = uploader
                meta.piano.title = title
                meta.piano.ytid = ytid
                meta.piano.duration = duration
                OmegaConf.save(meta, os.path.join(output_dir, ytid + ".yaml"))
                shutil.move(
                    os.path.join(tmpdir, filename),
                    os.path.join(output_dir, f"{ytid}.wav"),
                )

    return result


def download_pop(piano_id, pop_id, output_dir, dry_run):
    output_file_template = "%(id)s___%(title)s___%(duration)d.%(ext)s"
    pop_output_dir = os.path.join(output_dir, piano_id)
    os.makedirs(pop_output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, piano_id, output_file_template)
    url = f"https://www.youtube.com/watch?v={pop_id}"

    cmd = YTDLP + [
        "-o", output_template,
        "--extract-audio",
        "--audio-quality", "0",
        "--audio-format", "wav",
        "--retries", "25",
        "--match-filter", "duration < 300 & duration > 150",
        "--postprocessor-args", "ffmpeg:-ac 2 -ar 44100",
    ]
    if dry_run:
        cmd += ["--get-filename"]
    cmd.append(url)
    result = subprocess.call(cmd)

    if not dry_run:
        for filename in os.listdir(pop_output_dir):
            parsed = _parse_song_filename(filename)
            if parsed is None:
                continue
            ytid, title, duration = parsed
            yaml = os.path.join(output_dir, piano_id + ".yaml")

            meta = OmegaConf.load(yaml)
            meta.song = OmegaConf.create()
            meta.song.ytid = ytid
            meta.song.title = title
            meta.song.duration = duration

            OmegaConf.save(meta, yaml)
            shutil.move(
                os.path.join(pop_output_dir, filename),
                os.path.join(pop_output_dir, f"{ytid}.wav"),
            )


def download_pair(piano_id, pop_id, output_dir, dry_run=False):
    """Download one piano/song pair into one numbered dataset directory."""
    dataset_dir = next_dataset_root(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)

    download_piano(
        url=f"https://www.youtube.com/watch?v={piano_id}",
        output_dir=str(dataset_dir),
        postprocess=True,
        dry_run=dry_run,
    )
    piano_metadata = dataset_dir / f"{piano_id}.yaml"
    if not dry_run and not piano_metadata.exists():
        print(f"Skipping {piano_id}: piano download failed")
        shutil.rmtree(dataset_dir)
        return None

    download_pop(
        piano_id=piano_id,
        pop_id=pop_id,
        output_dir=str(dataset_dir),
        dry_run=dry_run,
    )
    if not dry_run:
        metadata = OmegaConf.load(piano_metadata)
        if not getattr(metadata, "song", None) or not metadata.song.get("ytid"):
            print(f"Skipping {piano_id}: song download failed")
            shutil.rmtree(dataset_dir)
            return None
    return dataset_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="piano cover downloader")

    parser.add_argument("dataset", type=str, default=None, help="provided csv")
    parser.add_argument("output_dir", type=str, default=None, help="output dir")
    parser.add_argument(
        "--num_audio",
        type=int,
        default=None,
        help="if specified, only {num_audio} pairs will be downloaded",
    )
    parser.add_argument(
        "--dry_run", default=False, action="store_true", help="whether dry_run"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    df = df[: args.num_audio]
    for piano_id, pop_id in tqdm(
        zip(df["piano_ids"], df["pop_ids"]),
        total=len(df),
    ):
        download_pair(piano_id, pop_id, args.output_dir, args.dry_run)
