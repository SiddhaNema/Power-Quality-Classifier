"""
demo.py
=======
Power Quality Disturbance Classifier — Presentation Demo

The input signal (300 samples / 3 cycles) is resampled to 100 samples
before inference to match the training data. Accuracy is unaffected.

Usage:
  python demo.py              # interactive menu
  python demo.py --demo       # cycle through all 17 signals
  python demo.py --signal ltspice_schematics/Sag.txt

Controls in demo mode:
  ENTER   next       p  previous       r  repeat
  #N      jump       c  pick by class  q  quit
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert

# ============================================================
#  CONFIG — must match training
# ============================================================
MODEL_PATH   = "models/best_model.pth"
FS           = 5000
F0           = 50
TRAIN_LEN    = 100     # samples the model was trained on (1 cycle)
WINDOW_SIZE  = 40
STRIDE       = 10
HIDDEN_DIM   = 64
DROPOUT      = 0.0
IN_CHANNELS  = 4
SIGNAL_DIR   = "ltspice_schematics"

CLASS_NAMES = [
    "Flicker", "Flicker_with_Sag", "Flicker_with_Swell",
    "Harmonics", "Harmonics_with_Sag", "Harmonics_with_Swell",
    "Interruption", "Notch", "Oscillatory_Transient",
    "Pure_Sinusoidal", "Sag", "Sag_with_Harmonics",
    "Sag_with_Oscillatory_Transient", "Swell", "Swell_with_Harmonics",
    "Swell_with_Oscillatory_Transient", "Transient"
]

NO_FAULT = {"Pure_Sinusoidal"}

SEVERITY = {
    "Pure_Sinusoidal":                  ("NORMAL",   "\033[92m"),
    "Sag":                              ("WARNING",  "\033[93m"),
    "Swell":                            ("WARNING",  "\033[93m"),
    "Interruption":                     ("CRITICAL", "\033[91m"),
    "Transient":                        ("SEVERE",   "\033[91m"),
    "Oscillatory_Transient":            ("WARNING",  "\033[93m"),
    "Harmonics":                        ("MODERATE", "\033[93m"),
    "Harmonics_with_Sag":               ("WARNING",  "\033[93m"),
    "Harmonics_with_Swell":             ("WARNING",  "\033[93m"),
    "Flicker":                          ("MODERATE", "\033[93m"),
    "Flicker_with_Sag":                 ("WARNING",  "\033[93m"),
    "Flicker_with_Swell":               ("WARNING",  "\033[93m"),
    "Sag_with_Oscillatory_Transient":   ("SEVERE",   "\033[91m"),
    "Swell_with_Oscillatory_Transient": ("SEVERE",   "\033[91m"),
    "Sag_with_Harmonics":               ("WARNING",  "\033[93m"),
    "Swell_with_Harmonics":             ("WARNING",  "\033[93m"),
    "Notch":                            ("MODERATE", "\033[93m"),
}

# Terminal colour codes
R  = "\033[0m"
B  = "\033[1m"
CY = "\033[96m"
DM = "\033[2m"


# ============================================================
#  MODEL  (identical architecture to training)
# ============================================================
class InstanceEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout):
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
        self.attn    = nn.Sequential(
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
        H  = self.encoder(bags.view(B * N, C, L)).view(B, N, -1)
        A  = self.attn(H).squeeze(-1)
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A  = torch.softmax(A, dim=1)
        out = self.classifier((A.unsqueeze(-1) * H).sum(dim=1))
        return (out, A) if return_attention else out


# ============================================================
#  FEATURE EXTRACTION  (identical to training)
# ============================================================
def extract_channels(sig):
    """sig: (100,) float32  →  (4, 100) float32"""
    L   = len(sig)
    sig = sig.astype(np.float32)

    mu, sd = sig.mean(), sig.std() + 1e-8
    ch0    = (sig - mu) / sd

    fft_mag = np.abs(np.fft.rfft(sig))
    fft_mag = fft_mag / (fft_mag.max() + 1e-8)
    ch1     = np.interp(np.linspace(0, len(fft_mag)-1, L),
                        np.arange(len(fft_mag)), fft_mag).astype(np.float32)

    freqs      = np.fft.rfftfreq(L, d=1/FS)
    harm_freqs = np.arange(F0, FS//2, F0)
    harm_mags  = np.array([
        fft_mag[np.argmin(np.abs(freqs - hf))] for hf in harm_freqs
    ])
    ch2 = np.interp(np.linspace(0, len(harm_mags)-1, L),
                    np.arange(len(harm_mags)), harm_mags).astype(np.float32)

    env = np.abs(hilbert(sig)).astype(np.float32)
    env = (env - env.mean()) / (env.std() + 1e-8)

    return np.stack([ch0, ch1, ch2, env], axis=0)   # (4, 100)


def make_bag(sig_100):
    """sig_100: (100,) → bag tensor (N_windows, 4, WINDOW_SIZE)"""
    ch = extract_channels(sig_100)
    L  = ch.shape[1]
    windows = [
        ch[:, s : s + WINDOW_SIZE]
        for s in range(0, L - WINDOW_SIZE + 1, STRIDE)
    ]
    return torch.tensor(np.array(windows), dtype=torch.float32)


# ============================================================
#  LOAD MODEL
# ============================================================
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ConjunctiveMIL(IN_CHANNELS, HIDDEN_DIM, len(CLASS_NAMES), DROPOUT).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"\n{B}ERROR:{R} Model not found at {MODEL_PATH!r}")
        print("Copy best_model.pth from Colab into the models/ folder.")
        sys.exit(1)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"{DM}Model loaded  [{device}]{R}\n")
    return model, device


# ============================================================
#  READ WAVEFORM
#  Reads the 300-sample .txt, resamples to 100 for inference.
#  The full 300-sample version is kept for plotting.
# ============================================================
def read_waveform(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd
        vals = pd.read_csv(path, header=None).values.flatten().astype(np.float32)
    else:
        with open(path) as f:
            lines = f.read().splitlines()
        data_lines = [l for l in lines
                      if l.strip() and not l.lower().startswith("time")]
        vals = []
        for line in data_lines:
            parts = line.split()
            try:
                vals.append(float(parts[-1]))
            except ValueError:
                continue
        vals = np.array(vals, dtype=np.float32)

    if len(vals) == 0:
        raise ValueError(f"No data found in {path!r}")

    mx = np.abs(vals).max()
    if mx > 0:
        vals /= mx

    # Full-length signal for plotting
    sig_full = vals.copy()

    # Resample to TRAIN_LEN for inference
    sig_infer = np.interp(
        np.linspace(0, len(vals)-1, TRAIN_LEN),
        np.arange(len(vals)), vals
    ).astype(np.float32)

    return sig_full, sig_infer


# ============================================================
#  INFERENCE
# ============================================================
@torch.no_grad()
def predict(model, device, sig_infer):
    bag    = make_bag(sig_infer).unsqueeze(0).to(device)
    mask   = torch.ones(1, bag.shape[1], device=device)
    out, A = model(bag, mask, return_attention=True)
    probs  = torch.softmax(out, dim=1).squeeze().cpu().numpy()
    pred_i = int(probs.argmax())
    return CLASS_NAMES[pred_i], float(probs[pred_i])*100, probs, A.squeeze().cpu().numpy()


# ============================================================
#  TERMINAL OUTPUT
# ============================================================
def print_result(pred_class, confidence, probs, attn_w, source=""):
    sev_label, sev_col = SEVERITY.get(pred_class, ("UNKNOWN", R))
    top5 = np.argsort(probs)[::-1][:5]
    W    = 66
    is_normal = pred_class in NO_FAULT

    print("\n" + "=" * W)
    print(f"  {B}{CY}SOURCE     :{R} {os.path.basename(source)}")
    print(f"  {B}{CY}PREDICTION :{R} {B}{pred_class}{R}")
    print(f"  {B}{CY}CONFIDENCE :{R} {B}{confidence:.1f}%{R}")
    print(f"  {B}{CY}SEVERITY   :{R} {sev_col}{B}{sev_label}{R}")
    print("-" * W)
    print(f"  TOP 5:")
    for i in top5:
        bar  = "█" * int(probs[i]*30) + "░" * (30 - int(probs[i]*30))
        flag = f"  {B}←{R}" if i == top5[0] else ""
        print(f"    {CLASS_NAMES[i]:<38} {bar} {probs[i]*100:5.1f}%{flag}")
    print("-" * W)

    if is_normal:
        print(f"  {B}FAULT LOCATION :{R} {sev_col}No fault — signal is healthy{R}")
    else:
        top_w  = int(attn_w.argmax())
        cs, ce = top_w * STRIDE, top_w * STRIDE + WINDOW_SIZE
        print(f"  {B}FAULT LOCATION :{R} samples {cs}–{ce}  "
              f"({cs/FS*1000:.1f} ms – {ce/FS*1000:.1f} ms)")
        print(f"\n  {B}ATTENTION:{R}")
        for i, w in enumerate(attn_w):
            s_i = i * STRIDE
            e_i = s_i + WINDOW_SIZE
            bar = "█" * int(w*24) + "░" * (24 - int(w*24))
            mk  = f"  {sev_col}{B}◄ FAULT{R}" if i == top_w else ""
            print(f"    W{i+1:02d} [{s_i:03d}–{e_i:03d}]  {bar}  {w:.3f}{mk}")

        step = max(1, TRAIN_LEN // 60)
        tl   = ["█" if cs <= k < ce else "─" for k in range(TRAIN_LEN)]
        comp = "".join(tl[i] for i in range(0, TRAIN_LEN, step))
        print(f"\n  TIMELINE  |{comp}|")

    print("=" * W)


# ============================================================
#  MATPLOTLIB PLOT  (shown, not saved)
# ============================================================
def show_plot(sig_full, sig_infer, pred_class, confidence, probs, attn_w, source=""):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    is_normal  = pred_class in NO_FAULT
    sev_label, _ = SEVERITY.get(pred_class, ("UNKNOWN", ""))
    top_w      = int(attn_w.argmax())
    cs, ce     = top_w * STRIDE, top_w * STRIDE + WINDOW_SIZE
    n_windows  = len(attn_w)

    # Full signal timeline
    t_full_ms = np.linspace(0, len(sig_full) / FS * 1000, len(sig_full))
    # Map fault region from infer-space to full-signal-space
    scale      = len(sig_full) / TRAIN_LEN
    cs_f, ce_f = int(cs * scale), int(ce * scale)
    t_cs_ms    = t_full_ms[cs_f]
    t_ce_ms    = t_full_ms[min(ce_f, len(t_full_ms)-1)]

    BG    = "#0d0d1a"
    PANEL = "#12122a"
    BLUE  = "#00cfff"
    RED   = "#ff4560"
    GREEN = "#00e396"
    AMBER = "#feb019"
    GRAY  = "#7777aa"
    WH    = "#dde0ff"

    sev_c = {"NORMAL": GREEN, "MODERATE": AMBER,
              "WARNING": AMBER, "SEVERE": RED, "CRITICAL": RED
              }.get(sev_label, AMBER)

    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.suptitle("Power Quality Disturbance Monitoring System",
                 color=WH, fontsize=14, fontweight="bold",
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.50, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.91, bottom=0.08)

    def _ax(pos, title, colspan=1):
        if colspan == 3:
            ax = fig.add_subplot(gs[pos[0], :])
        elif colspan == 2:
            ax = fig.add_subplot(gs[pos[0], pos[1]:pos[1]+2])
        else:
            ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=GRAY, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#22224a")
        ax.set_title(title, color=BLUE, fontsize=9,
                     fontweight="bold", pad=5, fontfamily="monospace")
        ax.grid(True, color="#1a1a3a", linewidth=0.4, linestyle="--")
        return ax

    # ── 1. Waveform (full width, top row) ──────────────────────────
    ax1 = _ax((0, 0), f"WAVEFORM  ·  {os.path.basename(source)}", colspan=3)
    ax1.plot(t_full_ms, sig_full, color=BLUE, linewidth=1.5, zorder=3)
    ax1.axhline(0, color=GRAY, linewidth=0.4, zorder=1)

    if not is_normal:
        ax1.axvspan(t_cs_ms, t_ce_ms, alpha=0.22, color=RED, zorder=2)
        ax1.axvline(t_cs_ms, color=RED, linewidth=1.0, linestyle="--", zorder=4)
        ax1.axvline(t_ce_ms, color=RED, linewidth=1.0, linestyle="--", zorder=4)

    # Info box
    box = (f"  {pred_class}\n"
           f"  Confidence : {confidence:.1f}%\n"
           f"  Severity   : {sev_label}  ")
    ax1.text(0.01, 0.97, box, transform=ax1.transAxes,
             fontsize=8, color=sev_c, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG,
                       edgecolor=sev_c, alpha=0.92))

    ax1.set_xlabel("Time (ms)", color=GRAY, fontsize=8)
    ax1.set_ylabel("Amplitude (pu)", color=GRAY, fontsize=8)
    ax1.set_xlim(t_full_ms[0], t_full_ms[-1])

    # ── 2. FFT spectrum ─────────────────────────────────────────────
    ax2 = _ax((1, 0), "FREQUENCY SPECTRUM")
    freqs   = np.fft.rfftfreq(len(sig_full), d=1/FS)
    fft_mag = np.abs(np.fft.rfft(sig_full))
    fft_db  = 20 * np.log10(fft_mag / (fft_mag.max()+1e-8) + 1e-8)
    mask_f  = freqs <= 800
    ax2.fill_between(freqs[mask_f], fft_db[mask_f],
                     fft_db[mask_f].min(), color=BLUE, alpha=0.45)
    ax2.plot(freqs[mask_f], fft_db[mask_f], color=BLUE, linewidth=1.0)
    for h in range(1, 9):
        hf = h * F0
        if hf <= 800:
            ax2.axvline(hf, color=AMBER, linewidth=0.5,
                        linestyle=":", alpha=0.6)
    ax2.set_xlabel("Frequency (Hz)", color=GRAY, fontsize=8)
    ax2.set_ylabel("Magnitude (dB)", color=GRAY, fontsize=8)

    # ── 3. Amplitude envelope ───────────────────────────────────────
    ax3 = _ax((1, 1), "AMPLITUDE ENVELOPE")
    from scipy.signal import hilbert as hbt
    env = np.abs(hbt(sig_full))
    ax3.plot(t_full_ms, np.abs(sig_full), color=GRAY,
             linewidth=0.7, alpha=0.45, label="|signal|")
    ax3.plot(t_full_ms, env, color=AMBER, linewidth=1.4, label="envelope")
    ax3.axhline(1.0, color=GREEN, linewidth=0.7, linestyle="--", alpha=0.55)
    if not is_normal:
        ax3.axvspan(t_cs_ms, t_ce_ms, alpha=0.15, color=RED)
    ax3.set_xlabel("Time (ms)", color=GRAY, fontsize=8)
    ax3.set_ylabel("Amplitude (pu)", color=GRAY, fontsize=8)
    ax3.legend(facecolor=PANEL, labelcolor=WH, fontsize=7,
               framealpha=0.7, loc="upper right")
    ax3.set_xlim(t_full_ms[0], t_full_ms[-1])

    # ── 4. Attention weights ────────────────────────────────────────
    ax4 = _ax((1, 2), "MIL ATTENTION  (fault location)")
    if is_normal:
        bar_cols = [GREEN] * n_windows
        ax4.set_title("MIL ATTENTION  ·  no fault", color=GREEN,
                      fontsize=9, fontweight="bold", pad=5,
                      fontfamily="monospace")
    else:
        bar_cols = [RED if i == top_w else BLUE for i in range(n_windows)]

    bars = ax4.bar(range(1, n_windows+1), attn_w,
                   color=bar_cols, edgecolor="#22224a", width=0.7)
    if not is_normal:
        bars[top_w].set_edgecolor(RED)
        bars[top_w].set_linewidth(2.0)

    ax4.set_xlabel("Window #", color=GRAY, fontsize=8)
    ax4.set_ylabel("Weight", color=GRAY, fontsize=8)
    step = max(1, n_windows // 10)
    ax4.set_xticks(range(1, n_windows+1, step))

    plt.show()


# ============================================================
#  RUN ONE FILE
# ============================================================
def run_single(path, model, device):
    sig_full, sig_infer = read_waveform(path)
    pred_class, confidence, probs, attn_w = predict(model, device, sig_infer)
    print_result(pred_class, confidence, probs, attn_w, source=path)
    show_plot(sig_full, sig_infer, pred_class, confidence, probs, attn_w, source=path)
    return pred_class, confidence


# ============================================================
#  DEMO MODE — persistent, never exits unless user types q
# ============================================================
def run_demo(model, device):
    files = sorted(glob.glob(os.path.join(SIGNAL_DIR, "*.txt")))
    if not files:
        print(f"No .txt files in {SIGNAL_DIR}/")
        print("Run ltspice_signals.py first.")
        return

    n   = len(files)
    idx = 0

    print(f"\n{B}{CY}{'='*66}{R}")
    print(f"  {B}POWER QUALITY CLASSIFIER  —  DEMO MODE  ({n} signals){R}")
    print(f"{B}{CY}{'='*66}{R}")
    print(f"  {CY}ENTER{R} next   {CY}p{R} prev   {CY}r{R} repeat")
    print(f"  {CY}#N{R}    jump   {CY}c{R} class  {CY}q{R} quit\n")

    while True:
        path = files[idx]
        print(f"\n{DM}[{idx+1}/{n}]  {os.path.basename(path)}{R}")
        run_single(path, model, device)

        # Keep prompting until a valid navigation command is given
        while True:
            print(f"\n  {DM}[{idx+1}/{n}] {os.path.basename(files[idx])}{R}")
            print(f"  {CY}ENTER{R}=next  {CY}p{R}=prev  {CY}r{R}=repeat  "
                  f"{CY}#N{R}=jump  {CY}c{R}=class  {CY}q{R}=quit")
            raw = input("  > ").strip().lower()

            if raw == "q":
                print(f"\n{CY}Demo ended.{R}\n")
                return

            elif raw in ("", "n"):          # next
                idx = (idx + 1) % n
                break

            elif raw == "p":                # previous
                idx = (idx - 1) % n
                break

            elif raw == "r":                # repeat
                break

            elif raw == "c":                # pick class
                class_names = sorted(set(
                    os.path.basename(f).split(".")[0] for f in files
                ))
                print(f"\n  {B}Classes:{R}")
                for ci, cn in enumerate(class_names, 1):
                    print(f"    {ci:2d}. {cn}")
                pick = input("  Name or number: ").strip()
                if pick.isdigit():
                    pi = int(pick) - 1
                    if 0 <= pi < len(class_names):
                        pick = class_names[pi]
                matches = [i for i, f in enumerate(files)
                           if os.path.basename(f).startswith(pick)]
                if matches:
                    idx = matches[0]
                    break
                else:
                    print(f"  No match for '{pick}'")

            elif raw.lstrip("#").isdigit(): # jump
                j = int(raw.lstrip("#")) - 1
                if 0 <= j < n:
                    idx = j
                    break
                else:
                    print(f"  Enter 1–{n}")

            else:                           # anything else → next
                idx = (idx + 1) % n
                break


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(SIGNAL_DIR, "*.txt")))

    model, device = load_model()

    print(f"{B}Available signals:{R}")
    for i, f in enumerate(files, 1):
        print(f"  {CY}{i:2d}{R}. {os.path.basename(f)}")

    print()
    raw = input("  Enter number or filename: ").strip()

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(files):
            run_single(files[idx], model, device)
        else:
            print(f"  Invalid — enter 1 to {len(files)}")
    else:
        # Try matching by name
        matches = [f for f in files if raw.lower() in os.path.basename(f).lower()]
        if len(matches) == 1:
            run_single(matches[0], model, device)
        elif len(matches) > 1:
            print(f"  Multiple matches:")
            for m in matches:
                print(f"    {os.path.basename(m)}")
        else:
            # Try as direct path
            if os.path.exists(raw):
                run_single(raw, model, device)
            else:
                print(f"  No match for '{raw}'")