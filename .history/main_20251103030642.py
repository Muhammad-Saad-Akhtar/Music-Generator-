# lab1_music_rnn.py
# MIT Intro to Deep Learning — Lab 1 (Part 2)
# Music Generation with an LSTM character model over ABC notation

import os
import sys
import argparse
import subprocess
import importlib
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Utilities: install/import pkgs
# -------------------------------
def _ensure_pkg(mod_name, pip_name=None, quiet=True):
    pip_name = pip_name or mod_name
    try:
        return importlib.import_module(mod_name)
    except ImportError:
        print(f"[info] Installing '{pip_name}'...")
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.append("--quiet")
        cmd.append(pip_name)
        try:
            subprocess.check_call(cmd)
            return importlib.import_module(mod_name)
        except Exception as e:
            print(f"[warn] Could not install '{pip_name}': {e}")
            return None

# Core deps
tf = _ensure_pkg("tensorflow")
np = _ensure_pkg("numpy")
tqdm = _ensure_pkg("tqdm")
scipy_io = _ensure_pkg("scipy.io")
if tf is None or np is None:
    sys.exit("[fatal] TensorFlow and NumPy are required.")

# Optional deps
comet_ml = _ensure_pkg("comet_ml", "comet_ml")
mdl = _ensure_pkg("mitdeeplearning", "mitdeeplearning")
IPy = _ensure_pkg("IPython")
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
except Exception:
    pass

write_wav = None
if scipy_io is not None and hasattr(scipy_io, "wavfile"):
    write_wav = scipy_io.wavfile.write

# -------------------------------------
# Comet (optional) — set your API key
# -------------------------------------
COMET_API_KEY = os.environ.get("VpaCw6OFk5oCMJnzGZviD3CP2", "")  

def create_experiment(params):
    if comet_ml is None or not COMET_API_KEY:
        print("[info] Comet disabled (missing package or API key).")
        return None
    exp = comet_ml.Experiment(api_key=COMET_API_KEY, project_name="6S191_Lab1_Part2")
    for k, v in params.items():
        exp.log_parameter(k, v)
    exp.flush()
    return exp

# -------------------------------
# Data loading / preprocessing
# -------------------------------
def load_abc_songs():
    if mdl is None:
        sys.exit("[fatal] 'mitdeeplearning' is required to load the dataset.")
    print("[info] Loading ABC songs...")
    songs = mdl.lab1.load_training_data()
    print(f"[info] Loaded {len(songs)} songs.")
    return songs

def build_vocab_and_maps(songs):
    joined = "\n\n".join(songs)
    vocab = sorted(set(joined))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    print(f"[info] Vocabulary size: {len(vocab)}")
    return joined, vocab, char2idx, idx2char

def vectorize_string(s, char2idx):
    return np.array([char2idx[c] for c in s])

def get_batch(vectorized, seq_length, batch_size):
    n = vectorized.shape[0] - 1
    idx = np.random.choice(n - seq_length, batch_size)
    input_batch  = [vectorized[i: i + seq_length] for i in idx]
    output_batch = [vectorized[i + 1: i + seq_length + 1] for i in idx]
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

# -------------------------------
# Model definition
# -------------------------------
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer="glorot_uniform",
        recurrent_activation="sigmoid",
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units),
        tf.keras.layers.Dense(vocab_size),
    ])

def compute_loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# -------------------------------
# Training
# -------------------------------
@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tf.reduce_mean(loss)

def train(model, optimizer, vectorized_songs, params, vocab_size, checkpoint_prefix, experiment=None):
    history = []
    pbar = tqdm.tqdm(range(params["num_training_iterations"]), desc="Training") if tqdm else range(params["num_training_iterations"])
    for it in pbar:
        x_b, y_b = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
        loss = train_step(model, optimizer, x_b, y_b).numpy()
        history.append(loss)
        if experiment is not None:
            experiment.log_metric("loss", loss, step=it)
        if tqdm:
            pbar.set_postfix(loss=f"{loss:.3f}")
        if it % 100 == 0:
            model.save_weights(checkpoint_prefix)
    model.save_weights(checkpoint_prefix)
    if experiment is not None:
        experiment.flush()
    return history

# -------------------------------
# Generation
# -------------------------------
def rebuild_for_inference(vocab_size, embedding_dim, rnn_units, checkpoint_prefix):
    # Batch size = 1 for stateful inference
    infer_model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    infer_model.build(tf.TensorShape([1, None]))
    infer_model.load_weights(checkpoint_prefix)
    infer_model.reset_states()
    return infer_model

def generate_text(model, start_string, char2idx, idx2char, length=1000):
    input_eval = [char2idx.get(s, 0) for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    rng = range(length)
    if tqdm: rng = tqdm.tqdm(rng, desc="Generating")
    for _ in rng:
        preds = model(input_eval)
        preds = tf.squeeze(preds, 0)
        predicted_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + "".join(text_generated)

def maybe_write_audio_assets(generated_text, experiment=None):
    if mdl is None:
        return
    songs = mdl.lab1.extract_song_snippet(generated_text)
    for i, song in enumerate(songs):
        waveform = mdl.lab1.play_song(song)  # may require abcmidi/timidity
        if waveform:
            print(f"[info] Generated song {i} (valid ABC).")
            # Save WAV if scipy present
            if write_wav is not None:
                numeric = np.frombuffer(waveform.data, dtype=np.int16)
                wav_path = f"output_{i}.wav"
                write_wav(wav_path, 88200, numeric)
                print(f"[info] Saved {wav_path}")
                if experiment is not None:
                    experiment.log_asset(wav_path)

# -------------------------------
# CLI / Main
# -------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Music generation with LSTM over ABC notation")
    ap.add_argument("--iters", type=int, default=3000, help="Training iterations")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--seq", type=int, default=100, help="Sequence length")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--embed", type=int, default=256, help="Embedding dim")
    ap.add_argument("--rnn", type=int, default=1024, help="LSTM units")
    ap.add_argument("--start", type=str, default="X", help="Start string for generation")
    ap.add_argument("--genlen", type=int, default=1000, help="Generated length")
    ap.add_argument("--ckptdir", type=str, default="./training_checkpoints", help="Checkpoint directory")
    ap.add_argument("--noaudio", action="store_true", help="Skip audio synthesis and saving")
    return ap.parse_args()

def main():
    args = parse_args()
    params = dict(
        num_training_iterations=args.iters,
        batch_size=args.batch,
        seq_length=args.seq,
        learning_rate=args.lr,
        embedding_dim=args.embed,
        rnn_units=args.rnn,
        gen_length=args.genlen,
        start=args.start,
    )

    os.makedirs(args.ckptdir, exist_ok=True)
    checkpoint_prefix = os.path.join(args.ckptdir, "my_ckpt.weights.h5")

    # Load & preprocess
    songs = load_abc_songs()
    joined, vocab, char2idx, idx2char = build_vocab_and_maps(songs)
    vectorized = vectorize_string(joined, char2idx)
    vocab_size = len(vocab)

    # Build/train
    model = build_model(vocab_size, args.embed, args.rnn, args.batch)
    model.build(tf.TensorShape([args.batch, args.seq]))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(args.lr)
    experiment = create_experiment(params)

    print("[info] Starting training...")
    train(model, optimizer, vectorized, params, vocab_size, checkpoint_prefix, experiment)

    # Rebuild for inference (batch=1) and generate
    print("[info] Restoring for inference and generating text...")
    infer_model = rebuild_for_inference(vocab_size, args.embed, args.rnn, checkpoint_prefix)
    generated = generate_text(infer_model, args.start, char2idx, idx2char, length=args.genlen)

    abc_path = "generated_abc.txt"
    with open(abc_path, "w", encoding="utf-8") as f:
        f.write(generated)
    print(f"[info] Saved generated ABC to {abc_path}")

    # Optional: audio
    if not args.noaudio:
        try:
            maybe_write_audio_assets(generated, experiment)
        except Exception as e:
            print(f"[warn] Audio synthesis skipped or failed: {e}")

    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()
