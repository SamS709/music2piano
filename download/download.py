
"""
Resumable YouTube piano/song downloader.

Usage:
    python download/download.py train_dataset.csv data \
        --cookies-from-browser chrome \
        --num_audio 4000 \
        --sleep-between-pairs 0

Features:
    - Resumable downloads
    - Persistent unavailable-video tracking
    - YouTube rate-limit detection and immediate shutdown
    - Atomic metadata writes
    - No deletion of existing dataset files
    - Live yt-dlp output
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import shutil

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm


sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_roots import dataset_roots, next_dataset_root


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

YTDLP = [
    sys.executable,
    "-m",
    "yt_dlp",
    "--no-warnings",
    "--js-runtimes",
    "deno",
]

DEFAULT_SLEEP_REQUESTS = 0
DEFAULT_SLEEP_MIN = 0
DEFAULT_SLEEP_MAX = 0


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------

class RateLimitError(RuntimeError):
    """YouTube has rate-limited the current session."""


class UnavailableVideoError(RuntimeError):
    """The requested video is unavailable."""


# ---------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------

def _ytdlp_command(cookies_from_browser=None, cookies=None):
    command = list(YTDLP)

    if cookies_from_browser:
        command += ["--cookies-from-browser", cookies_from_browser]

    if cookies:
        command += ["--cookies", cookies]

    command += [
        "--extractor-args",
        "youtube:player_client=mweb",
    ]

    return command


def _run_ytdlp(cmd):
    """
    Execute yt-dlp while displaying its output live.

    Stop immediately if YouTube reports rate limiting.
    """

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output = []

    rate_limit_patterns = [
        "rate-limited",
        "rate limited",
        "too many requests",
        "http error 429",
    ]

    unavailable_patterns = [
        "video unavailable",
        "private video",
        "this video is private",
        "has been removed",
        "video has been removed",
        "not available in your country",
        "this content isn't available",
    ]

    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)

            lower = line.lower()

            if any(pattern in lower for pattern in rate_limit_patterns):
                process.terminate()
                process.wait()

                raise RateLimitError(
                    "YouTube rate limit detected. "
                    "Stopping the download batch."
                )

        return_code = process.wait()

    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise

    full_output = "".join(output).lower()

    if any(pattern in full_output for pattern in unavailable_patterns):
        raise UnavailableVideoError(full_output)

    return return_code


# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------

def _existing_dataset_root(piano_id, output_dir):
    for candidate in dataset_roots(output_dir):
        if (candidate / f"{piano_id}.yaml").exists():
            return candidate

    return None


def _existing_pair_root(piano_id, pop_id, output_dir):
    """Return a dataset root only when this exact piano/song pair is complete."""
    for candidate in dataset_roots(output_dir):
        metadata_path = candidate / f"{piano_id}.yaml"
        if not metadata_path.exists():
            continue
        try:
            metadata = OmegaConf.load(metadata_path)
        except Exception:
            continue
        if metadata.get("piano", {}).get("ytid") != piano_id:
            continue
        if metadata.get("song", {}).get("ytid") != pop_id:
            continue
        if not (candidate / f"{piano_id}.wav").exists():
            continue
        if not (candidate / piano_id / f"{pop_id}.wav").exists():
            continue
        return candidate
    return None


def _find_existing_video(video_id, output_dir, kind):
    """Return an existing WAV and metadata for a piano or song video."""
    for root in dataset_roots(output_dir):
        candidates = (
            [root / f"{video_id}.wav"]
            if kind == "piano"
            else root.rglob(f"{video_id}.wav")
        )
        for audio_path in candidates:
            metadata_path = (
                audio_path.parent / f"{video_id}.yaml"
                if kind == "piano"
                else audio_path.parent.parent / f"{audio_path.parent.name}.yaml"
            )
            if not metadata_path.exists():
                continue
            metadata = OmegaConf.load(metadata_path)
            section = metadata.get(kind)
            if section is not None and section.get("ytid") == video_id:
                return audio_path, section
    return None


def _reuse_existing_video(video_id, output_dir, kind, metadata_path):
    existing = _find_existing_video(video_id, output_dir, kind)
    if existing is None:
        return False

    source_path, video_metadata = existing
    metadata = OmegaConf.load(metadata_path)
    destination_dir = metadata_path.parent
    if kind == "piano":
        shutil.copy2(source_path, destination_dir / f"{video_id}.wav")
        metadata.piano = video_metadata
    else:
        piano_dir = destination_dir / str(metadata.piano.ytid)
        piano_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, piano_dir / f"{video_id}.wav")
        metadata.song = video_metadata
    _save_metadata(metadata, metadata_path)
    return True


def _save_metadata(metadata, path):
    """Save metadata atomically to avoid corrupted YAML files."""

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    OmegaConf.save(metadata, tmp_path)
    os.replace(tmp_path, path)


def _is_complete_pair(piano_id, output_dir):
    dataset_dir = _existing_dataset_root(piano_id, output_dir)

    if dataset_dir is None:
        return False

    metadata_path = dataset_dir / f"{piano_id}.yaml"

    try:
        metadata = OmegaConf.load(metadata_path)

        if not metadata.get("song"):
            return False

        song_id = metadata.song.get("ytid")

        if not song_id:
            return False

        piano_id_in_metadata = metadata.piano.get("ytid")

        return (
            piano_id_in_metadata is not None
            and (dataset_dir / f"{piano_id}.wav").exists()
            and (dataset_dir / piano_id / f"{song_id}.wav").exists()
        )

    except Exception:
        return False


# ---------------------------------------------------------------------
# Failure tracking
# ---------------------------------------------------------------------

def _load_unavailable(output_dir):
    path = Path(output_dir) / "unavailable_videos.txt"

    if not path.exists():
        return set()

    with open(path, encoding="utf-8") as f:
        return {
            line.strip().split("\t", 1)[0]
            for line in f
            if line.strip()
        }


def _record_unavailable(output_dir, video_id, stage, reason):
    """
    Record permanently unavailable videos.

    The file format is:
        video_id    stage    reason
    """

    path = Path(output_dir) / "unavailable_videos.txt"

    with open(path, "a", encoding="utf-8") as f:
        reason = re.sub(r"\s+", " ", reason).strip()
        f.write(f"{video_id}\t{stage}\t{reason}\n")

    tqdm.write(
        f"Recorded unavailable video: {video_id} ({stage})"
    )


def _record_failure(output_dir, video_id, stage, reason):
    """Record other failures without permanently blacklisting the ID."""

    path = Path(output_dir) / "failed_videos.txt"

    with open(path, "a", encoding="utf-8") as f:
        reason = re.sub(r"\s+", " ", reason).strip()
        f.write(f"{video_id}\t{stage}\t{reason}\n")


# ---------------------------------------------------------------------
# Piano download
# ---------------------------------------------------------------------

def download_piano(
    url,
    output_dir,
    postprocess=True,
    dry_run=False,
    cookies_from_browser=None,
    cookies=None,
    sleep_requests=DEFAULT_SLEEP_REQUESTS,
    sleep_min=DEFAULT_SLEEP_MIN,
    sleep_max=DEFAULT_SLEEP_MAX,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        output = os.path.join(
            tmpdir,
            "%(uploader)s___%(title)s___%(id)s___%(duration)d.%(ext)s",
        )

        cmd = _ytdlp_command(cookies_from_browser, cookies) + [
            "-o", output,
            "-f", "bestaudio[ext=m4a]/bestaudio/best",
            "--extract-audio",
            "--audio-quality", "0",
            "--audio-format", "wav",
            "--retries", "5",
            "--force-ipv4",
            "--yes-playlist",
            "--sleep-requests", str(sleep_requests),
            "--sleep-interval", str(sleep_min),
            "--max-sleep-interval", str(sleep_max),
        ]

        if postprocess:
            cmd += [
                "--postprocessor-args",
                "ffmpeg:-ac 1 -ar 16000",
            ]

        if dry_run:
            cmd += ["--get-filename"]

        cmd.append(url)

        result = _run_ytdlp(cmd)

        if dry_run:
            return result

        for filename in os.listdir(tmpdir):

            parsed = _parse_piano_filename(filename)

            if parsed is None:
                continue

            uploader, title, ytid, duration = parsed

            metadata_path = output_dir / f"{ytid}.yaml"

            metadata = OmegaConf.create()
            metadata.piano = OmegaConf.create()

            metadata.piano.uploader = uploader
            metadata.piano.title = title
            metadata.piano.ytid = ytid
            metadata.piano.duration = duration

            # Preserve existing song metadata if present.
            if metadata_path.exists():
                old_metadata = OmegaConf.load(metadata_path)

                if old_metadata.get("song"):
                    metadata.song = old_metadata.song

            _save_metadata(metadata, metadata_path)

            source = Path(tmpdir) / filename
            destination = output_dir / f"{ytid}.wav"

            if not destination.exists():
                shutil.move(str(source), str(destination))

        return result


# ---------------------------------------------------------------------
# Song download
# ---------------------------------------------------------------------

def download_pop(
    piano_id,
    pop_id,
    output_dir,
    dry_run=False,
    cookies_from_browser=None,
    cookies=None,
    sleep_requests=DEFAULT_SLEEP_REQUESTS,
    sleep_min=DEFAULT_SLEEP_MIN,
    sleep_max=DEFAULT_SLEEP_MAX,
):
    output_dir = Path(output_dir)

    piano_dir = output_dir / piano_id
    piano_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(
        piano_dir / "%(id)s___%(title)s___%(duration)d.%(ext)s"
    )

    url = f"https://www.youtube.com/watch?v={pop_id}"

    cmd = _ytdlp_command(cookies_from_browser, cookies) + [
        "-o", output_template,
        "-f", "bestaudio[ext=m4a]/bestaudio/best",
        "--extract-audio",
        "--audio-quality", "0",
        "--audio-format", "wav",
        "--retries", "5",
        "--match-filter", "duration < 300 & duration > 150",
        "--postprocessor-args", "ffmpeg:-ac 2 -ar 44100",
        "--sleep-requests", str(sleep_requests),
        "--sleep-interval", str(sleep_min),
        "--max-sleep-interval", str(sleep_max),
    ]

    if dry_run:
        cmd += ["--get-filename"]

    cmd.append(url)

    result = _run_ytdlp(cmd)

    if dry_run:
        return result

    metadata_path = output_dir / f"{piano_id}.yaml"

    for filename in os.listdir(piano_dir):

        parsed = _parse_song_filename(filename)

        if parsed is None:
            continue

        ytid, title, duration = parsed

        metadata = OmegaConf.load(metadata_path)

        metadata.song = OmegaConf.create()
        metadata.song.ytid = ytid
        metadata.song.title = title
        metadata.song.duration = duration

        _save_metadata(metadata, metadata_path)

        source = piano_dir / filename
        destination = piano_dir / f"{ytid}.wav"

        if not destination.exists():
            shutil.move(str(source), str(destination))

    return result


# ---------------------------------------------------------------------
# Pair download
# ---------------------------------------------------------------------

def download_pair(
    piano_id,
    pop_id,
    output_dir,
    dry_run=False,
    cookies_from_browser=None,
    cookies=None,
    sleep_requests=DEFAULT_SLEEP_REQUESTS,
    sleep_min=DEFAULT_SLEEP_MIN,
    sleep_max=DEFAULT_SLEEP_MAX,
):
    """
    Download one piano/song pair.

    Existing files are preserved.
    Incomplete pairs can be resumed.
    """

    output_dir = Path(output_dir)

    existing_pair = _existing_pair_root(piano_id, pop_id, output_dir)
    if existing_pair is not None:
        return existing_pair

    dataset_dir = _existing_dataset_root(piano_id, output_dir)

    created_dir = False

    if dataset_dir is None:
        dataset_dir = next_dataset_root(output_dir)
        dataset_dir.mkdir(parents=True, exist_ok=False)
        created_dir = True

    metadata_path = dataset_dir / f"{piano_id}.yaml"
    piano_audio = dataset_dir / f"{piano_id}.wav"

    # -------------------------------------------------------------
    # Piano
    # -------------------------------------------------------------

    piano_complete = (
        metadata_path.exists() and piano_audio.exists()
    )

    if not piano_complete:

        try:
            reused = _reuse_existing_video(
                piano_id, output_dir, "piano", metadata_path
            )
            if not reused:
                download_piano(
                    url=f"https://www.youtube.com/watch?v={piano_id}",
                    output_dir=str(dataset_dir),
                    postprocess=True,
                    dry_run=dry_run,
                    cookies_from_browser=cookies_from_browser,
                    cookies=cookies,
                    sleep_requests=sleep_requests,
                    sleep_min=sleep_min,
                    sleep_max=sleep_max,
                )

        except UnavailableVideoError as exc:
            _record_unavailable(
                output_dir, piano_id, "piano", str(exc)
            )
            return None

        if not dry_run and not (
            metadata_path.exists() and piano_audio.exists()
        ):
            _record_failure(
                output_dir,
                piano_id,
                "piano",
                "Piano download failed or produced no WAV file",
            )

            if created_dir:
                try:
                    dataset_dir.rmdir()
                except OSError:
                    pass

            return None

    if dry_run:
        return dataset_dir

    # -------------------------------------------------------------
    # Song
    # -------------------------------------------------------------

    metadata = OmegaConf.load(metadata_path)
    song_id = metadata.song.get("ytid") if metadata.get("song") else None

    song_complete = (
        song_id == pop_id
        and (dataset_dir / piano_id / f"{song_id}.wav").exists()
    )

    if song_complete:
        return dataset_dir

    try:
        reused = _reuse_existing_video(
            pop_id, output_dir, "song", metadata_path
        )
        if not reused:
            download_pop(
                piano_id=piano_id,
                pop_id=pop_id,
                output_dir=str(dataset_dir),
                dry_run=False,
                cookies_from_browser=cookies_from_browser,
                cookies=cookies,
                sleep_requests=sleep_requests,
                sleep_min=sleep_min,
                sleep_max=sleep_max,
            )

    except UnavailableVideoError as exc:
        _record_unavailable(
            output_dir, pop_id, "song", str(exc)
        )
        return None

    metadata = OmegaConf.load(metadata_path)

    song_id = metadata.song.get("ytid") if metadata.get("song") else None

    if (
        song_id is None
        or not (dataset_dir / piano_id / f"{song_id}.wav").exists()
    ):
        _record_failure(
            output_dir,
            pop_id,
            "song",
            "Song download failed or produced no WAV file",
        )
        return None

    return dataset_dir


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Resumable piano cover downloader"
    )

    parser.add_argument("dataset", type=str)
    parser.add_argument("output_dir", type=str)

    parser.add_argument(
        "--num_audio",
        type=int,
        default=None,
        help="Maximum number of audios to download",
    )

    parser.add_argument(
        "--start_audio",
        type=int,
        default=0,
        help="Number of CSV rows to skip before downloading (default: 0)",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
    )

    parser.add_argument(
        "--sleep-between-pairs",
        type=float,
        default=0,
        help="Seconds to wait after each pair (default: 0)",
    )

    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=DEFAULT_SLEEP_REQUESTS,
        help="Seconds to wait between HTTP requests (default: 0)",
    )

    parser.add_argument(
        "--sleep-min",
        type=float,
        default=DEFAULT_SLEEP_MIN,
        help="Minimum yt-dlp download sleep in seconds (default: 0)",
    )

    parser.add_argument(
        "--sleep-max",
        type=float,
        default=DEFAULT_SLEEP_MAX,
        help="Maximum yt-dlp download sleep in seconds (default: 0)",
    )

    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser cookies, e.g. chrome or firefox:default-release",
    )

    parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape-format cookies.txt file",
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry IDs previously recorded as unavailable",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cookies and args.cookies_from_browser:
        parser.error(
            "--cookies and --cookies-from-browser are mutually exclusive"
        )

    if args.start_audio < 0:
        parser.error("--start_audio must be >= 0")

    if (
        args.sleep_requests < 0
        or args.sleep_min < 0
        or args.sleep_max < args.sleep_min
        or args.sleep_between_pairs < 0
    ):
        parser.error(
            "--sleep-requests, --sleep-min, and "
            "--sleep-between-pairs must be >= 0; "
            "--sleep-max must be >= --sleep-min"
        )

    df = pd.read_csv(args.dataset)

    required_columns = {"piano_ids", "pop_ids"}

    if not required_columns.issubset(df.columns):
        parser.error(
            f"CSV must contain columns: {required_columns}"
        )

    df = df.iloc[args.start_audio:]

    if args.num_audio is not None:
        df = df.head(args.num_audio)

    unavailable = (
        set() if args.retry_failed
        else _load_unavailable(output_dir)
    )

    completed = 0
    skipped = 0
    failed = 0

    try:

        for piano_id, pop_id in tqdm(
            zip(df["piano_ids"], df["pop_ids"]),
            total=len(df),
            desc="Downloading",
            dynamic_ncols=True,
        ):

            piano_id = str(piano_id).strip()
            pop_id = str(pop_id).strip()

            if not piano_id or not pop_id:
                skipped += 1
                continue

            if piano_id in unavailable or pop_id in unavailable:
                skipped += 1
                continue

            if not args.dry_run and _is_complete_pair(
                piano_id, output_dir
            ):
                skipped += 1
                continue

            try:

                result = download_pair(
                    piano_id=piano_id,
                    pop_id=pop_id,
                    output_dir=str(output_dir),
                    dry_run=args.dry_run,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies=args.cookies,
                    sleep_requests=args.sleep_requests,
                    sleep_min=args.sleep_min,
                    sleep_max=args.sleep_max,
                )

                if result is None:
                    failed += 1
                else:
                    completed += 1

            except RateLimitError as exc:

                tqdm.write("\n" + "=" * 65)
                tqdm.write("YOUTUBE RATE LIMIT DETECTED")
                tqdm.write(str(exc))
                tqdm.write(
                    "Stopping safely. Restart the script after "
                    "the restriction has expired."
                )
                tqdm.write("=" * 65)

                break

            if args.sleep_between_pairs > 0:
                time.sleep(args.sleep_between_pairs)

    except KeyboardInterrupt:
        tqdm.write("\nDownload interrupted by user.")
        tqdm.write("Completed downloads have been preserved.")

    finally:

        tqdm.write("\n" + "=" * 45)
        tqdm.write("DOWNLOAD SUMMARY")
        tqdm.write("=" * 45)
        tqdm.write(f"Completed: {completed}")
        tqdm.write(f"Skipped:   {skipped}")
        tqdm.write(f"Failed:    {failed}")
        tqdm.write("=" * 45)


if __name__ == "__main__":
    main()
