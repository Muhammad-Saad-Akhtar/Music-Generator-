import os
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
        x = self.embed(x)              # (B,T,E)
        out, hidden = self.lstm(x, hidden)  # (B,T,H)
        logits = self.fc(out)          # (B,T,V)
        return logits, hidden

@torch.no_grad()
def generate_text(model, start_string, char2idx, idx2char, length=1000, temperature=0.9, device=None):
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed to ids
    seed_ids = [char2idx.get(ch, 0) for ch in start_string]
    x = torch.tensor([seed_ids], dtype=torch.long, device=device)  # (1,T0)
    hidden = None
    # warm up
    _, hidden = model(x, hidden)

    last_id = x[0, -1].view(1, 1)  # (1,1)
    out_chars = list(start_string)

    for _ in range(length):
        logits, hidden = model(last_id, hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)
        last_id = next_id
        out_chars.append(idx2char[next_id.item()])

    return "".join(out_chars)

def build_vocab_and_maps(songs):
    joined = "\n\n".join(songs)
    vocab = sorted(set(joined))         
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = np.array(vocab)
    return joined, vocab, char2idx, idx2char

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training_checkpoints/torch_lstm.pt", help="Path to weights")
    ap.add_argument("--embed", type=int, default=256, help="Embedding dim used in training")
    ap.add_argument("--rnn", type=int, default=512, help="LSTM hidden units used in training")
    ap.add_argument("--layers", type=int, default=1, help="LSTM layers used in training")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout used in training (if layers>1)")
    ap.add_argument("--start", type=str, default="X", help="Start string")
    ap.add_argument("--len", type=int, default=1000, help="Characters to generate")
    ap.add_argument("--temp", type=float, default=0.9, help="Sampling temperature")
    ap.add_argument("--out", type=str, default="generated_abc.txt", help="Output ABC file")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        import mitdeeplearning as mdl
    except Exception:
        raise SystemExit("Please `pip install mitdeeplearning` to rebuild the training vocab.")

    songs = mdl.lab1.load_training_data()
    _, vocab, char2idx, idx2char = build_vocab_and_maps(songs)
    vocab_size = len(vocab)

    # Build model + load weights
    model = CharLSTM(vocab_size, args.embed, args.rnn, num_layers=args.layers, dropout=args.dropout).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)

    # Generate
    text = generate_text(
        model, start_string=args.start, char2idx=char2idx, idx2char=idx2char,
        length=args.len, temperature=args.temp, device=device
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[ok] Wrote {args.out}")

if __name__ == "__main__":
    main()
