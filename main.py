import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert

# ============================================================
#  CONFIG
# ============================================================
CONFIG = {
    "seed": 42,

    # Paths
    "data_dir":  "/content/Dataset",
    "model_dir": "/content/models",
    "model_name": "best_model.pth",

    # Signal properties
    "fs":          5000,   # sampling rate (Hz)
    "f0":          50,     # fundamental frequency (Hz)
    "signal_len":  100,    # samples per signal

    # Augmentation
    "num_augmentations": 5,
    "aug_noise_std":     0.02,
    "aug_scale_range":   (0.95, 1.05),
    "aug_shift_max":     5,

    # Windowing
    "window_size": 50,
    "stride":      25,

    # Model
    "hidden_dim":  64,
    "dropout":     0.3,

    # Training
    "num_epochs":  100,
    "batch_size":  32,
    "lr":          1e-3,
    "weight_decay": 1e-4,
    "grad_clip":   1.0,
    "label_smoothing": 0.05,
    "scheduler_step":  15,
    "scheduler_gamma": 0.5,
    "patience": 10,

    # Split
    "test_size": 0.20,
    "val_size":  0.20,
}

CLASS_NAMES = [
    "Pure_Sinusoidal", "Sag", "Swell", "Interruption", "Transient",
    "Oscillatory_Transient", "Harmonics", "Harmonics_with_Sag",
    "Harmonics_with_Swell", "Flicker", "Flicker_with_Sag",
    "Flicker_with_Swell", "Sag_with_Oscillatory_Transient",
    "Swell_with_Oscillatory_Transient", "Sag_with_Harmonics",
    "Swell_with_Harmonics", "Notch"
]

# ============================================================
#  SETUP
# ============================================================
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

os.makedirs(CONFIG["model_dir"], exist_ok=True)

# ============================================================
#  FEATURE EXTRACTION
#  Each signal becomes 4 channels:
#    ch0 — raw time-domain (normalised)
#    ch1 — full FFT magnitude spectrum
#    ch2 — harmonic energy profile (bins at 50,150,250,350,450 Hz interpolated)
#    ch3 — instantaneous envelope via Hilbert transform
# ============================================================
def extract_channels(sig, fs, f0, L):
    """sig: (L,) float32  →  out: (4, L) float32"""
    # ch0: z-score normalised time domain
    mu, sd = sig.mean(), sig.std() + 1e-8
    ch0 = (sig - mu) / sd

    # ch1: FFT magnitude, mirrored back to length L
    fft_mag = np.abs(np.fft.rfft(sig))                  # (L//2+1,)
    fft_mag = fft_mag / (fft_mag.max() + 1e-8)
    # Interpolate to length L so all channels have same size
    ch1 = np.interp(np.linspace(0, len(fft_mag)-1, L),
                    np.arange(len(fft_mag)), fft_mag).astype(np.float32)

    # ch2: harmonic energy profile
    # For each sample position, map to the nearest harmonic bin energy
    # This gives the model a direct view of harmonic content
    freqs    = np.fft.rfftfreq(L, d=1/fs)
    harm_freqs = np.arange(f0, fs//2, f0)              # 50,100,150,...,2500
    harm_mags  = np.array([
        fft_mag[np.argmin(np.abs(freqs - hf))]
        for hf in harm_freqs
    ])                                                   # (50,)
    ch2 = np.interp(np.linspace(0, len(harm_mags)-1, L),
                    np.arange(len(harm_mags)), harm_mags).astype(np.float32)

    # ch3: instantaneous amplitude envelope via analytic signal
    analytic  = hilbert(sig)
    envelope  = np.abs(analytic).astype(np.float32)
    envelope  = (envelope - envelope.mean()) / (envelope.std() + 1e-8)

    return np.stack([ch0, ch1, ch2, envelope], axis=0)  # (4, L)


# ============================================================
#  DATA LOADING — from CSV files
# ============================================================
def load_from_csv(data_dir, fs, f0):
    csv_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".csv")
    ]
    print(f"CSV files found : {len(csv_files)}")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {data_dir!r}.")

    signals, labels, label_names = [], [], []
    unique_labels = sorted([os.path.splitext(os.path.basename(f))[0] for f in csv_files])
    label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}

    for path in csv_files:
        df = pd.read_csv(path, header=None)
        if df.shape[0] < df.shape[1]:
            df = df.T
        label_name = os.path.splitext(os.path.basename(path))[0]
        label_idx  = label_map[label_name]
        L          = df.shape[1]
        for row in df.values.astype(np.float32):
            channels = extract_channels(row, fs, f0, L)
            signals.append(channels)
            labels.append(label_idx)

    print(f"Loaded {len(signals)} signals × {signals[0].shape[0]} channels × {signals[0].shape[1]} samples")
    print(f"Classes : {unique_labels}")
    return signals, np.array(labels, dtype=np.int64), unique_labels


# ============================================================
#  AUGMENTATION
# ============================================================
def augment(signals, labels, cfg):
    aug_signals, aug_labels = [], []
    lo, hi    = cfg["aug_scale_range"]
    noise_std = cfg["aug_noise_std"]
    shift_max = cfg["aug_shift_max"]

    for sig, lbl in zip(signals, labels):
        aug_signals.append(sig)
        aug_labels.append(lbl)
        for _ in range(cfg["num_augmentations"]):
            noise  = np.random.normal(0, noise_std, sig.shape).astype(np.float32)
            scale  = np.random.uniform(lo, hi)
            shift  = np.random.randint(-shift_max, shift_max + 1)
            new_sig = np.roll(sig * scale + noise, shift, axis=1)
            aug_signals.append(new_sig)
            aug_labels.append(lbl)

    return aug_signals, np.array(aug_labels, dtype=np.int64)


# ============================================================
#  BAG CREATION
# ============================================================
def create_bags(signals, window_size, stride):
    bags = []
    for sig in signals:                      # sig: (C, L)
        L = sig.shape[1]
        windows = [
            sig[:, s : s + window_size]
            for s in range(0, L - window_size + 1, stride)
        ]
        if not windows:
            windows = [sig]
        bags.append(torch.tensor(np.array(windows), dtype=torch.float32))
    return bags


# ============================================================
#  MODEL
# ============================================================
class InstanceEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.drop(F.relu(self.fc(x)))


class ConjunctiveMIL(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes, dropout):
        super().__init__()
        self.encoder = InstanceEncoder(in_channels, hidden_dim, dropout)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, bags, mask=None, return_attention=False):
        B, N, C, L = bags.shape
        H = self.encoder(bags.view(B * N, C, L)).view(B, N, -1)
        A = self.attn(H).squeeze(-1)
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A = torch.softmax(A, dim=1)
        bag_rep = (A.unsqueeze(-1) * H).sum(dim=1)
        out = self.classifier(bag_rep)
        return (out, A) if return_attention else out


# ============================================================
#  COLLATE
# ============================================================
def collate(batch_bags, batch_labels):
    lengths = [b.shape[0] for b in batch_bags]
    padded  = pad_sequence(batch_bags, batch_first=True)
    mask    = torch.zeros(padded.shape[0], padded.shape[1])
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    return padded.to(device), mask.to(device), batch_labels.to(device)


# ============================================================
#  EVALUATION
# ============================================================
@torch.no_grad()
def evaluate(model, bags, labels_tensor, batch_size):
    model.eval()
    correct = total = 0
    for i in range(0, len(bags), batch_size):
        b_bags, b_mask, b_labels = collate(bags[i : i + batch_size],
                                            labels_tensor[i : i + batch_size])
        preds   = torch.argmax(model(b_bags, b_mask), dim=1)
        correct += (preds == b_labels).sum().item()
        total   += len(b_labels)
    return correct / total


# ============================================================
#  CONFUSION MATRIX
# ============================================================
@torch.no_grad()
def print_confusion(model, bags, labels_tensor, class_names, batch_size):
    model.eval()
    all_preds, all_true = [], []
    for i in range(0, len(bags), batch_size):
        b_bags, b_mask, b_labels = collate(bags[i : i + batch_size],
                                            labels_tensor[i : i + batch_size])
        preds = torch.argmax(model(b_bags, b_mask), dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(b_labels.cpu().tolist())

    n      = len(class_names)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(all_true, all_preds):
        matrix[t][p] += 1

    print("\n" + "=" * 64)
    print("  PER-CLASS ACCURACY")
    print("=" * 64)
    for i, lbl in enumerate(class_names):
        total   = sum(matrix[i])
        correct = matrix[i][i]
        acc     = correct / total if total else 0
        bar     = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {lbl:<42} {bar} {acc*100:5.1f}%")

    print("\n" + "=" * 64)
    print("  TOP CONFUSIONS  (true → predicted)")
    print("=" * 64)
    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i][j] > 0:
                confusions.append((matrix[i][j], class_names[i], class_names[j]))
    for count, tl, pl in sorted(confusions, reverse=True)[:10]:
        print(f"  {tl:<38} → {pl:<38} ({count}x)")
    print("=" * 64)


# ============================================================
#  TRAINING
# ============================================================
def train(model, train_bags, train_labels, val_bags, val_labels, cfg, model_path):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"]
    )

    best_val_acc     = 0.0
    patience_counter = 0

    print("\n" + "=" * 64)
    print(f"  Training  |  epochs={cfg['num_epochs']}  batch={cfg['batch_size']}")
    print("=" * 64)

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = 0.0
        perm = np.random.permutation(len(train_bags))

        for i in range(0, len(train_bags), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            b_bags, b_mask, b_labels = collate(
                [train_bags[j] for j in idx], train_labels[idx]
            )
            optimizer.zero_grad()
            loss = criterion(model(b_bags, b_mask), b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        val_acc = evaluate(model, val_bags, val_labels, cfg["batch_size"])
        lr_now  = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:03d}/{cfg['num_epochs']} | "
              f"Loss {total_loss:8.4f} | Val Acc {val_acc:.4f} | LR {lr_now:.2e}", end="")

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            torch.save({"model_state": model.state_dict(),
                        "label_map":   {n: i for i, n in enumerate(class_names)},
                        "class_names": class_names,
                        "in_channels": 4}, model_path)
            print("  ✓ saved")
        else:
            patience_counter += 1
            print()

        if patience_counter >= cfg["patience"]:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    print(f"\n  Best validation accuracy : {best_val_acc:.4f}")


# ============================================================
#  DISTURBANCE LOCALISATION
# ============================================================
def locate_disturbance(model, sample_idx, bags, labels_tensor,
                       class_names, window_size, stride):
    model.eval()
    bag        = bags[sample_idx].to(device)
    true_label = class_names[labels_tensor[sample_idx].item()]

    with torch.no_grad():
        out, attn = model(bag.unsqueeze(0),
                          torch.ones(1, bag.shape[0], device=device),
                          return_attention=True)

    pred_label = class_names[torch.argmax(out, dim=1).item()]
    attn_w     = attn.squeeze().cpu().numpy()
    top_window = int(attn_w.argmax())

    print("\n" + "=" * 64)
    print("  DISTURBANCE LOCALISATION ANALYSIS")
    print("=" * 64)
    print(f"  True class      : {true_label}")
    print(f"  Predicted class : {pred_label}")
    print("-" * 64)

    for i, w in enumerate(attn_w):
        s, e   = i * stride, i * stride + window_size
        marker = " ◄ TOP" if i == top_window else ""
        print(f"  Window {i+1:02d}  [{s:03d}–{e:03d}]  weight={w:.4f}{marker}")

    cs, ce = top_window * stride, top_window * stride + window_size
    print("-" * 64)
    print(f"  Focus region : samples {cs}–{ce}")
    print(f"  Disturbance  : {pred_label}")
    sig_len = bags[sample_idx].shape[2]   # L from (N, C, L) — wait, bag is (N, C, W)
    # Use window_size * num_windows as proxy for original signal length
    timeline = ["-"] * 100
    for k in range(cs, min(ce, 100)):
        timeline[k] = "█"
    print(f"\n  Timeline : {''.join(timeline)}")
    print("=" * 64)


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    signals, labels, class_names = load_from_csv(
        CONFIG["data_dir"], CONFIG["fs"], CONFIG["f0"]
    )

    signals, labels = augment(signals, labels, CONFIG)
    print(f"Samples after augmentation : {len(signals)}")

    bags          = create_bags(signals, CONFIG["window_size"], CONFIG["stride"])
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    idx = np.arange(len(bags))
    train_idx, test_idx = train_test_split(
        idx, test_size=CONFIG["test_size"],
        stratify=labels_tensor.numpy(), random_state=CONFIG["seed"]
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=CONFIG["val_size"],
        stratify=labels_tensor.numpy()[train_idx], random_state=CONFIG["seed"]
    )

    train_bags,  train_labels = [bags[i] for i in train_idx], labels_tensor[train_idx]
    val_bags,    val_labels   = [bags[i] for i in val_idx],   labels_tensor[val_idx]
    test_bags,   test_labels  = [bags[i] for i in test_idx],  labels_tensor[test_idx]

    in_channels = signals[0].shape[0]   # 4
    num_classes = len(CLASS_NAMES)

    model = ConjunctiveMIL(
        in_channels=in_channels,
        hidden_dim=CONFIG["hidden_dim"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"],
    ).to(device)

    model_path = os.path.join(CONFIG["model_dir"], CONFIG["model_name"])

    train(model, train_bags, train_labels, val_bags, val_labels, CONFIG, model_path)

    print("\nLoading best checkpoint…")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])

    test_acc = evaluate(model, test_bags, test_labels, CONFIG["batch_size"])
    print(f"\n{'=' * 64}")
    print(f"  Final Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"{'=' * 64}")

    print_confusion(model, test_bags, test_labels, class_names, CONFIG["batch_size"])

    rng_idx = test_idx[np.random.randint(len(test_idx))]
    locate_disturbance(
        model, rng_idx, bags, labels_tensor, class_names,
        CONFIG["window_size"], CONFIG["stride"]
    )