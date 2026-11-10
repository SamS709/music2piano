import glob
import os
import random
import sys

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midiaudiopair import MidiAudioPair
from data_roots import metadata_files


def split_spleeter(meta_files, device="auto"):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Use audio loader explicitly for loading audio waveform :
    from spleeter.audio.adapter import AudioAdapter
    from spleeter.separator import Separator
    import spleeter
    import tensorflow as tf

    gpu_devices = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpu_devices:
        raise RuntimeError(
            "GPU was requested, but TensorFlow cannot see a CUDA GPU. "
            "Install the TensorFlow CUDA dependencies for the Spleeter environment."
        )
    if device == "cpu":
        print("Spleeter device: CPU")
    elif gpu_devices:
        print(f"Spleeter device: GPU ({gpu_devices[0].name})")
    else:
        print("Spleeter device: CPU (no TensorFlow GPU is available)")

    sample_rate = 44100
    audio_loader = AudioAdapter.default()

    separator = None

    for meta_file in tqdm(meta_files):
        sample = MidiAudioPair(meta_file)
        if sample.error_code == MidiAudioPair.NO_SONG:
            continue
        if os.path.exists(sample.vocals):
            continue
        if not os.path.isfile(sample.song):
            print(f"Skipping {meta_file}: aligned song is missing ({sample.song})")
            continue

        if separator is None:
            # Using embedded configuration.
            separator = Separator("spleeter:2stems")
        waveform, _ = audio_loader.load(sample.song, sample_rate=sample_rate)

        # Perform the separation :
        prediction = separator.separate(waveform)

        audio_loader.save(
            path=sample.vocals,
            data=prediction["vocals"][:, 0:1],
            codec=spleeter.audio.Codec.MP3,
            sample_rate=sample_rate,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="bpm estimate using essentia")

    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="""directory contains {id}/{pop_filename.wav}
        """,
    )

    parser.add_argument(
        "--random_order",
        default=False,
        action="store_true",
        help="Random order process (to run multiple process)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="TensorFlow device: auto (default), cpu, or gpu",
    )

    args = parser.parse_args()

    meta_files = [str(path) for path in metadata_files(args.data_dir)]
    if args.random_order:
        random.shuffle(meta_files)

    print("meta ", len(meta_files))

    split_spleeter(meta_files, device=args.device)
