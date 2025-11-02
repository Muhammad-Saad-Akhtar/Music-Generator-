````markdown
# Music Generator (PyTorch, ABC Notation)

This repository provides a complete pipeline for training and generating music in **ABC notation** using a character-level **LSTM** model implemented in **PyTorch**.  
It consists of two main scripts:

- **`main.py`** — trains the model and generates ABC tunes.
- **`generator.py`** — loads saved weights and generates new tunes without retraining.

The generated ABC text can be converted to playable audio (MIDI/WAV) using **WSL (Windows Subsystem for Linux)** tools.

---

## Table of Contents
1. [Overview](#1-overview)  
2. [File Descriptions](#2-file-descriptions)  
3. [Outputs](#3-outputs)  
4. [Converting to Audio in WSL](#4-converting-to-audio-in-wsl)  
5. [Installation](#5-installation)  
6. [Usage Examples](#6-usage-examples)  
7. [Troubleshooting](#7-troubleshooting)  
8. [Configuration and Reproducibility](#8-configuration-and-reproducibility)  
9. [License and Notes](#9-license-and-notes)  
10. [Summary](#10-summary)

---

## 1. Overview

### What is ABC Notation?
ABC notation is a simple text-based format for writing music, commonly used for folk tunes. Each note, bar, and key signature is represented as a character.

### Workflow
1. Load a dataset of folk songs in ABC format.  
2. Build a character vocabulary.  
3. Train a character-level LSTM to predict the next character.  
4. Generate new ABC music text.  
5. Convert the text into MIDI and WAV for playback.

---

## 2. File Descriptions

### **`main.py`**
**Purpose:** Train a character-level LSTM and generate a tune.

#### Key Components
1. **Data loading**
   - Uses the `mitdeeplearning` package to load ABC songs.
   - Creates mappings between characters and indices.

2. **Model definition**
   - `CharLSTM`: An LSTM-based model with embedding and linear output layers.

3. **Training loop**
   - Trains using batches of text.
   - Uses cross-entropy loss and Adam optimizer.
   - Periodically saves weights to `training_checkpoints/torch_lstm.pt`.

4. **Generation loop**
   - After training, generates new ABC text character-by-character.

5. **Optional Comet ML integration**
   - Logs training metrics if a valid API key is provided.

6. **Outputs**
   - `torch_lstm.pt`: Model weights.
   - `generated_abc.txt`: Generated ABC notation.
   - `output_*.wav`: Optional synthesized audio (if supported).

#### Example command
```bash
python main.py --iters 3000 --batch 4 --seq 100 --lr 5e-3 --embed 256 --rnn 512 --genlen 1200 --temp 0.9
````

---

### **`generator.py`**

**Purpose:** Generate new music from saved weights (no retraining).

#### Key Components

1. Loads a configuration file (`model_config.json`) if available.
2. Rebuilds the vocabulary using the same dataset.
3. Constructs the model and loads trained weights.
4. Generates a sequence of characters (ABC music).
5. Saves the generated tune to `generated_abc.txt` (or a specified output path).

#### Example command

```bash
python generator.py --ckpt training_checkpoints/torch_lstm.pt --start "X" --len 1200 --temp 0.85 --out new_tune.abc
```

If `model_config.json` exists:

```bash
python generator.py --ckpt training_checkpoints/torch_lstm.pt
```

---

## 3. Outputs

| File                                 | Description                           |
| ------------------------------------ | ------------------------------------- |
| `training_checkpoints/torch_lstm.pt` | Trained PyTorch model weights         |
| `generated_abc.txt`                  | Generated ABC music text              |
| `song.mid`                           | MIDI file (after conversion via WSL)  |
| `song.wav`                           | Audio file (after conversion via WSL) |

Example ABC snippet:

```
X:1
T:Generated Tune
M:4/4
K:D
D2 FA d2 | A2 BA F2 E2 |
```

---

## 4. Converting to Audio in WSL

### Step 1: Install tools

Open Ubuntu (WSL) and run:

```bash
sudo apt update
sudo apt install -y abcmidi timidity
```

### Step 2: Navigate to the folder

In WSL:

```bash
cd "/mnt/c/Users/HP/Desktop/Others/Music-Generator-"
```

### Step 3: Convert ABC → MIDI

```bash
abc2midi "generated_abc.txt" -o song.mid
```

### Step 4: Convert MIDI → WAV

```bash
timidity "song.mid" -Ow -o "song.wav"
```

### Step 5: Play audio directly in WSL

```bash
timidity "song.mid"
```

The output files (`song.mid` and `song.wav`) will appear in the same Windows folder:

```
C:\Users\HP\Desktop\Others\Music-Generator-\
```

---

## 5. Installation

### Python dependencies

```bash
pip install torch numpy mitdeeplearning scipy
```

### Optional (for Comet ML tracking)

```bash
pip install comet_ml
```

### GPU setup

* The scripts automatically detect CUDA if available.
* For limited VRAM, use smaller hyperparameters:

  * `--batch 2–4`
  * `--rnn 256–512`
  * `--seq 96–128`

---

## 6. Usage Examples

### Train a new model

```bash
python main.py --iters 3000 --batch 8 --seq 100 --lr 5e-3 --embed 256 --rnn 512
```

### Generate new music

```bash
python generator.py --ckpt training_checkpoints/torch_lstm.pt --len 1200 --temp 0.9
```

### Convert and play in WSL

```bash
cd "/mnt/c/Users/HP/Desktop/Others/Music-Generator-"
abc2midi "generated_abc.txt" -o song.mid
timidity "song.mid" -Ow -o song.wav
timidity "song.mid"    # play directly
```

---

## 7. Troubleshooting

**Size mismatch error**
Ensure model architecture matches training. Either:

* Use the same CLI parameters (`--rnn`, `--embed`, `--layers`, etc.), or
* Use the `model_config.json` created during training.

**No audio playback**

* Check that WSL2 is enabled.
* Ensure `abcmidi` and `timidity` are installed.
* Use `timidity song.mid` to test playback.

**Unstructured or short tunes**

* Increase training iterations (`--iters`).
* Reduce temperature (`--temp 0.8`).
* Start with a full ABC header (`--start "X:1\nT:AI Tune\nM:4/4\nK:D\n"`).

---

## 8. Configuration and Reproducibility

To automatically store model parameters for later generation, add this snippet to the end of `main.py` after training:

```python
import json, os
cfg = {
  "embedding_dim": args.embed,
  "rnn_units": args.rnn,
  "num_layers": args.layers,
  "dropout": args.dropout,
}
with open(os.path.join(args.ckptdir, "model_config.json"), "w") as f:
    json.dump(cfg, f)
```

`generator.py` will detect this file and load it automatically.

---

## 9. License and Notes

* The training dataset is loaded via `mitdeeplearning.lab1`, which includes public-domain Irish folk music for educational purposes.
* Generated outputs are original compositions based on learned statistical structures.
* Cite MIT Introduction to Deep Learning if used for research or derivative work.

---

## 10. Summary

| Script         | Function                                     | Key Output                           |
| -------------- | -------------------------------------------- | ------------------------------------ |
| `main.py`      | Trains LSTM on ABC tunes and generates music | `torch_lstm.pt`, `generated_abc.txt` |
| `generator.py` | Uses trained weights to create new music     | `generated_abc.txt`                  |
| WSL commands   | Converts ABC → MIDI → WAV for playback       | `song.mid`, `song.wav`               |

```
```
