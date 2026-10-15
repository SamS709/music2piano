# Installation:

2 environments are required for the data:

- The first to fetch youtube:

```bash
conda create -n ytdlp python=3.11 -y
conda activate ytdlp
python -m pip install -U "yt-dlp[default]"
conda install -c conda-forge ffmpeg nodejs -y
python -m pip install pandas tqdm joblib omegaconf
```

- The second to run the following scripts:

```bash
conda create -n music2piano python=3.9
pip install -r requirements.txt
```


# Preprocess Scripts

1. transform wav to midi 
```bash
python wav_to_midi.py DATA_DIR
```

2. synchronize midi 
```bash
python pop_align.py DATA_DIR
```

3. Estimate Pop's beats
```bash
python bpm_quantize.py DATA_DIR 
```

4. get separated vocal track
```bash
python split_spleeter.py DATA_DIR
```

5. caculate melody chroma accuracy
```bash
python melody_accuracy.py DATA_DIR 
```

# Expected Structure
```
├── -7lV0oJ0QXc
│   ├── EHl_eQhgefw.beatinterval.npy
│   ├── EHl_eQhgefw.beatstep.npy
│   ├── EHl_eQhgefw.beattime.npy
│   ├── EHl_eQhgefw.mid
│   ├── EHl_eQhgefw.notes.npy
│   ├── EHl_eQhgefw.pitchshift.wav
│   ├── EHl_eQhgefw.qmidi.mid
│   ├── EHl_eQhgefw.qmix.flac
│   ├── EHl_eQhgefw.vocals.mp3
│   ├── EHl_eQhgefw.wav
│   └── The Beatles - With a Little Help from My Friends ____With A Little Help From My Friends - The Beatles _.txt
├── -7lV0oJ0QXc.mid
├── -7lV0oJ0QXc.wav
├── -7lV0oJ0QXc.yaml
```

## Descriptions for each data
1. ```*.beattime.npy```
    - timesteps (unit : second) extracted using essentia. ```np.ndarray```. (num_beats, )
2. ```*.beatstep.npy```
    - timesteps (unit : second) per every half-beat. it is calculated using linear interpolation of ```beattime```.
2. ```*.notes.npy```
    - ```np.ndarray``` shape: ```(number_of_notes, 4)```
    - each row contains : ```[onset(unit: index), offset(unit: index), pitch, velocity]```
    - onset/offset values mean that the index of ```beatstep``` time. 
    - for example, 
        - ```beatstep = [0.6, 1.0, 1.4]```
        - ```note = [0, 1, 77, 88]``
        - then ```note``` means a note starts from 0.6sec to 1.0sec, and its pitch is 77 and velocity is 88.

