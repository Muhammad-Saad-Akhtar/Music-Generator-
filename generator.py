import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Model Definition
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_layers=1, dropout=0.0):
        super().__init__()
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
        x = self.embed(x)                   # (B, T, E)
        out, hidden = self.lstm(x, hidden)  # (B, T, H)
        logits = self.fc(out)               # (B, T, V)
        return logits, hidden

@torch.no_grad()
def generate_text(model, start_string, char2idx, idx2char, length=1000, temperature=0.9, device=None):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed -> ids
    seed_ids = [char2idx.get(ch, 0) for ch in start_string]
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)  # (1, T0)
    hidden = None
    _, hidden = model(x, hidden)  # warm up hidden with the seed

    last_id = x[0, -1].view(1, 1)
    out_chars = list(start_string)

    for _ in range(length):
        logits, hidden = model(last_id, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        last_id = next_id
        out_chars.append(idx2char[next_id.item()])

    return "".join(out_chars)

def build_vocab_and_maps():
    # Rebuild the SAME vocab ordering used in training
    import mitdeeplearning as mdl
    songs = mdl.lab1.load_training_data()
    joined = "\n\n".join(songs)
    vocab = sorted(set(joined))
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = np.array(vocab)
    return vocab, char2idx, idx2char

def load_model_config(cfg_path):
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def parse_args():
    ap = argparse.ArgumentParser(description="Generate ABC from saved weights")
    ap.add_argument("--ckpt", default="training_checkpoints/torch_lstm.pt", help="Path to .pt weights")
    ap.add_argument("--config", default="training_checkpoints/model_config.json", help="Path to model config JSON")
    # Fallbacks if config is missing:
    ap.add_argument("--embed", type=int, default=256, help="Embedding dim (fallback)")
    ap.add_argument("--rnn", type=int, default=1024, help="LSTM hidden units (fallback)")
    ap.add_argument("--layers", type=int, default=1, help="LSTM layers (fallback)")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout (fallback)")
    # Generation controls:
    ap.add_argument("--start", type=str, default="X", help="Seed string")
    ap.add_argument("--len", type=int, default=1200, help="Characters to generate")
    ap.add_argument("--temp", type=float, default=0.9, help="Sampling temperature")
    ap.add_argument("--out", type=str, default="generated_abc.txt", help="Output ABC file")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model config if available 
    cfg = load_model_config(args.config)
    if cfg:
        emb = int(cfg.get("embedding_dim", args.embed))
        rnn = int(cfg.get("rnn_units", args.rnn))
        layers = int(cfg.get("num_layers", args.layers))
        dropout = float(cfg.get("dropout", args.dropout))
        print(f"[info] Loaded config from {args.config} -> embed={emb}, rnn={rnn}, layers={layers}, dropout={dropout}")
    else:
        emb, rnn, layers, dropout = args.embed, args.rnn, args.layers, args.dropout
        print(f"[warn] No config found at {args.config}. Using CLI: embed={emb}, rnn={rnn}, layers={layers}, dropout={dropout}")

    try:
        vocab, char2idx, idx2char = build_vocab_and_maps()
    except Exception as e:
        raise SystemExit(f"[fatal] Could not rebuild vocab (install mitdeeplearning). Error: {e}")
    vocab_size = len(vocab)
    print(f"[info] Vocab size: {vocab_size}")

    # Build model and load weights
    model = CharLSTM(vocab_size, emb, rnn, num_layers=layers, dropout=dropout).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)  # will error if arch doesn't match (which is what we want)
    print(f"[info] Loaded weights from {args.ckpt}")

    # Generate
    text = generate_text(
        model,
        start_string=args.start,
        char2idx=char2idx,
        idx2char=idx2char,
        length=args.len,
        temperature=args.temp,
        device=device,
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[ok] Wrote {args.out}")

if __name__ == "__main__":
    main()
