import glob, os, sys
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_roots import dataset_roots

def main(data_dir: str = "data"):

    transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)

    for root in dataset_roots(data_dir):
        for wav in glob.glob(f"{root}/*.wav"):          # top-level wavs = piano covers
            midi = wav.replace(".wav", ".mid")
            if os.path.exists(midi):
                continue
            audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
            transcriptor.transcribe(audio, midi)
        
    
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="piano cover downloader")

    parser.add_argument(
        "data_dir",
        type=str,
        default=None,
        help="""directory contains {id}/{song_filename.wav}
        """,
    )
  

    args = parser.parse_args()

    main(data_dir = args.data_dir)