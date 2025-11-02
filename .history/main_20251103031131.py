# lab1_music_rnn_torch.py
# MIT Intro to Deep Learning — Lab 1 (Part 2), PyTorch port
# Character-level LSTM over ABC notation for music generation

import os
import sys
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps
try:
    import mitdeeplearning as mdl   # dataset + ABC helpers
except Exception:
    mdl = None

try:
    import comet_ml                 # optional experiment tracking
except Exception:
    comet_ml = None

# ---------- Utils ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(s): print(s, flush=True)

# ---------- Data ----------

def load_abc_songs():
    if mdl is None:
        sys.exit("[fatal] 'mitdeeplearning' is required for the ABC dataset. pip install mitdeeplearning")
    log("[info] Loading ABC songs...")
    songs = mdl.lab1.load_training_data()
    log(f"[info] Loaded {len(songs)} songs.")
    return songs

def build_vocab_and_maps(songs):
    joined = "\n\n".join(songs)
    vocab = sorted(set(joined))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = np.array(vocab)
    log(f"[info] Vocabulary size = {len(vocab)}")
    return joined, vocab, char2idx, idx2char

def vectorize_string(s, char2idx):
    return np.array([char2idx[c] for c in s], dtype=np.int64)

def get_batch(vectorized, seq_len, batch_size):
    """
    Returns x, y with shapes:
      x: (batch, seq_len)
      y: (batch, seq_len) — next char targets
    """
    n = vectorized.shape[0] - 1
    idx = np.random.choice(n - seq_len, batch_size)
    x = [vectorized[i : i + seq_len] for i in idx]
    y = [vectorized[i + 1 : i + seq_len + 1] for i in idx]
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

# ---------- Model ----------

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_layers=1, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: LongTensor (batch, seq_len)
        hidden: optional (h0, c0)
        returns:
          logits: (batch, seq_len, vocab)
          hidden: (h, c)
        """
        emb = self.embed(x)              # (B, T, E)
        out, hidden = self.lstm(emb, hidden)  # (B, T, H)
        logits = self.fc(out)            # (B, T, V)
        return logits, hidden

# ---------- Training ----------

def train_loop(model, opt, loss_fn, vectorized, params, device, experiment=None):
    model.train()
    history = []
    iters = params["num_training_iterations"]
    seq_len = params["seq_length"]
    batch_size = params["batch_size"]
    ckpt_path = params["ckpt_path"]

    for it in range(iters):
        x_np, y_np = get_batch(vectorized, seq_len, batch_size)
        x = torch.tensor(x_np, dtype=torch.long, device=device)
        y = torch.tensor(y_np, dtype=torch.long, device=device)

        opt.zero_grad()
        logits, _ = model(x)  # (B, T, V)
        # reshape for CE: (B*T, V) vs (B*T)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_val = float(loss.item())
        history.append(loss_val)
        if experiment is not None:
            experiment.log_metric("loss", loss_val, step=it)

        if (it % 50) == 0 or it == iters - 1:
            log(f"[iter {it:05d}] loss={loss_val:.4f}")

        if (it % 200) == 0 and it > 0:
            torch.save(model.state_dict(), ckpt_path)

    torch.save(model.state_dict(), ckpt_path)
    return history

# ---------- Generation ----------

@torch.no_grad()
def generate_text(model, start_string, char2idx, idx2char, length=1000, device=None, temperature=1.0):
    """
    Autoregressive generation with multinomial sampling.
    """
    model.eval()
    device = device or device_auto()

    # seed → tensor
    input_ids = [char2idx.get(ch, 0) for ch in start_string]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)  # (1, T0)
    hidden = None
    generated = list(start_string)

    # Warm up hidden on the seed
    logits, hidden = model(x, hidden=hidden)

    # Next-token loop (continue one char at a time)
    last_id = x[0, -1].view(1, 1)  # (1,1)
    for _ in range(length):
        logits, hidden = model(last_id, hidden=hidden)    # logits: (1,1,V)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        last_id = next_id
        generated.append(idx2char[next_id.item()])

    return "".join(generated)

def maybe_audio_from_abc(generated_text, experiment=None):
    if mdl is None:
        log("[info] mitdeeplearning not installed; skipping audio synthesis.")
        return
    try:
        songs = mdl.lab1.extract_song_snippet(generated_text)
        for i, song in enumerate(songs):
            waveform = mdl.lab1.play_song(song)  # may require abcmidi/timidity
            if waveform:
                log(f"[info] Generated song {i} (valid ABC).")
                # Save wav if you want (scipy optional)
                try:
                    from scipy.io.wavfile import write
                    import numpy as np
                    numeric = np.frombuffer(waveform.data, dtype=np.int16)
                    wav_path = f"output_{i}.wav"
                    write(wav_path, 88200, numeric)
                    log(f"[info] Saved {wav_path}")
                    if experiment is not None:
                        experiment.log_asset(wav_path)
                except Exception as e:
                    log(f"[warn] Could not save WAV: {e}")
    except Exception as e:
        log(f"[warn] Audio synthesis failed or tools missing: {e}")

# ---------- Comet ----------

def init_comet(params):
    api_key = os.environ.get("COMET_API_KEY", "")
    if not api_key or comet_ml is None:
        log("[info] Comet disabled (no API key or package).")
        return None
    exp = comet_ml.Experiment(api_key=api_key, project_name="6S191_Lab1_Part2_PyTorch")
    for k, v in params.items():
        exp.log_parameter(k, v)
    exp.flush()
    return exp

# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser(description="PyTorch LSTM music generator (ABC notation)")
    ap.add_argument("--iters", type=int, default=3000, help="Training iterations")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--seq", type=int, default=100, help="Sequence length")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--embed", type=int, default=256, help="Embedding dim")
    ap.add_argument("--rnn", type=int, default=1024, help="LSTM hidden units")
    ap.add_argument("--layers", type=int, default=1, help="LSTM layers")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout between LSTM layers")
    ap.add_argument("--start", type=str, default="X", help="Start string for generation")
    ap.add_argument("--genlen", type=int, default=1000, help="Generated length")
    ap.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--ckptdir", type=str, default="./training_checkpoints", help="Checkpoint directory")
    ap.add_argument("--noaudio", action="store_true", help="Skip audio synthesis")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    return ap.parse_args()

# ---------- Main ----------

def main():
    args = parse_args()
    set_seed(args.seed)
    dev = device_auto()
    log(f"[info] Using device: {dev}")

    os.makedirs(args.ckptdir, exist_ok=True)
    ckpt_path = os.path.join(args.ckptdir, "torch_lstm.pt")

    # Hyperparams bag for logging
    params = dict(
        num_training_iterations=args.iters,
        batch_size=args.batch,
        seq_length=args.seq,
        learning_rate=args.lr,
        embedding_dim=args.embed,
        rnn_units=args.rnn,
        num_layers=args.layers,
        dropout=args.dropout,
        gen_length=args.genlen,
        start=args.start,
        temperature=args.temp,
        ckpt_path=ckpt_path,
    )

    # Data
    songs = load_abc_songs()
    joined, vocab, char2idx, idx2char = build_vocab_and_maps(songs)
    vectorized = vectorize_string(joined, char2idx)
    vocab_size = len(vocab)

    # Model
    model = CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embed,
        rnn_units=args.rnn,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(dev)

    # Optim/criterion
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Comet
    experiment = init_comet(params)

    # Train
    log("[info] Starting training...")
    train_loop(model, opt, loss_fn, vectorized, params, dev, experiment)

    # Generate
    log("[info] Generating text...")
    gen = generate_text(
        model, start_string=args.start, char2idx=char2idx, idx2char=idx2char,
        length=args.genlen, device=dev, temperature=args.temp
    )

    abc_path = "generated_abc.txt"
    with open(abc_path, "w", encoding="utf-8") as f:
        f.write(gen)
    log(f"[info] Saved generated ABC to {abc_path}")

    if not args.noaudio:
        maybe_audio_from_abc(gen, experiment)

    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()
