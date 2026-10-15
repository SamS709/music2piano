import glob, os
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

def main(data_dir: str = "data"):

    transcriptor = PianoTranscription(device="cpu", checkpoint_path=None)

    for wav in glob.glob(f"{data_dir}/*.wav"):          # top-level wavs = piano covers
        midi = wav.replace(".wav", ".mid")       # -> data/-7lV0oJ0QXc.mid
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