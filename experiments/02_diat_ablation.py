# ============================================================================
# A cross-domain double dissociation of neuron-task matching in pure
# spiking networks  --  Dikmen & Karadag
#
# Verified, locked experiment script. The body below is the exact code that
# produced the paper numbers; it is kept self-contained for Colab paste-and-run.
# Requires the dikmen-spiking-neurons library (NeuronRegistry). See README.
#
# Experiment 2/3 — HEADLINE: matched-capacity ablation on DIAT-uSAT
# Paper: Section 4.1 "Heterogeneity does not win" and Figure 3.
# Families at matched capacity: homogeneous (D0_*), vertical heterogeneous (D1_*),
# horizontal path bank (D2_concat). Doppler-LIF homogeneous is the matched baseline.
# Source cell: "Shd deep probe" notebook, cell 1.
# ============================================================================

"""
DIAT-uSAT DEEP ABLATION — pure-SNN MS-IF heterogeneity (vertical vs horizontal)
Experiment 2/3: HEADLINE. Micro-Doppler radar (spectrogram-as-sequence, T=64).

Config A (accuracy-first): pure-SNN (inter-layer signal is spikes; no BatchNorm/
LayerNorm, no softmax gate). Per-type threshold calibration ONCE at init to avoid
dead/saturated layers; firing then runs free and is logged as an efficiency metric
(a sparsity-controlled variant is a separate secondary study). Capacity-matched,
paired bootstrap vs the best homogeneous baseline, pre-registered before results.

Paste into one Colab cell and Run. Resumes from checkpoint if interrupted.
"""

# ════════════════════════════════════════════════════════════════
# Imports + environment
# ════════════════════════════════════════════════════════════════
import os, sys, json, time, copy
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt   # Colab handles backend; never matplotlib.use('Agg')

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    ON_COLAB = True
except Exception:
    ON_COLAB = False

RESEARCH = Path('/content/drive/MyDrive/Research') if ON_COLAB else Path('.')
RESEARCH.mkdir(parents=True, exist_ok=True)
PROJECT = RESEARCH / 'NISAC_DeepHetero'
PROJECT.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ════════════════════════════════════════════════════════════════
# Config (no argparse) + experiment manifest
# ════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # architecture
    L: int = 4
    W: int = 128
    K: int = 5
    target_firing: float = 0.15
    stft_window: int = 16             # DIAT T=64 -> window <= T
    norm: str = "none"                # PURE SNN: no normalization layer
    # training (early stopping on validation: train to each config's own plateau)
    max_epochs: int = 200
    patience: int = 20                # stop after this many epochs with no smoothed-val improvement
    val_frac: float = 0.15            # validation carved from train (model selection ONLY)
    val_smooth: int = 5               # moving-average window on val score (denoise epoch selection)
    select_metric: str = "macro_f1"   # primary metric for early stopping + headline ("accuracy" | "macro_f1")
    lr: float = 1e-3
    batch_size: int = 128
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    # firing band (logged as a diagnostic; NOT enforced during training under config A)
    fire_lo: float = 0.02
    fire_hi: float = 0.90
    # methods to run
    methods: list = field(default_factory=lambda: [
        "D0_doppler", "D0_stft", "D0_chirp", "D0_dualtau",
        "D1_H1", "D1_rev", "D1_perm1", "D2_concat"])
    # experiment manifest (single source of truth)
    exp_index: int = 2
    exp_total: int = 3
    exp_name: str = "DIAT-uSAT deep ablation (HEADLINE)"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "SHD deep probe (T=12)", "done"),
        (2, "DIAT-uSAT deep ablation (D0/D1/perm/D2)", "THIS"),
        (3, "cross-family + sparsity-energy curve", "pending"),
    ])

cfg = Config()

METHOD_LABELS = {
    "D0_doppler": "D0 homog Doppler-LIF",
    "D0_stft":    "D0 homog STFT-IF",
    "D0_chirp":   "D0 homog Chirp-LIF",
    "D0_dualtau": "D0 homog Dual-tau-LIF",
    "D1_H1":      "D1 vertical (STFT->Doppler->Chirp->Dualtau)",
    "D1_rev":     "D1 vertical reversed (falsification)",
    "D1_perm1":   "D1 vertical permutation-1 (falsification)",
    "D2_concat":  "D2 horizontal bank (concat)",
}
H1_ORDER = ["STFT-IF", "Doppler-LIF", "Chirp-LIF", "Dual-tau-LIF"]
H1_REV   = ["Dual-tau-LIF", "Chirp-LIF", "Doppler-LIF", "STFT-IF"]
D1_PERM1 = ["Doppler-LIF", "Dual-tau-LIF", "STFT-IF", "Chirp-LIF"]
D2_PATHS = ["Doppler-LIF", "Chirp-LIF", "STFT-IF", "Dual-tau-LIF", "CrossInhib-LIF"]


# ════════════════════════════════════════════════════════════════
# dikmen-spiking-neurons  (pip-install from GitHub if not present)
# ════════════════════════════════════════════════════════════════
def _clear_dikmen():
    for k in [k for k in sys.modules if k.startswith('dikmen')]:
        del sys.modules[k]

_clear_dikmen()
try:
    import dikmen_neurons  # noqa: F401  (already installed?)
except ImportError:
    import subprocess
    url = "git+https://github.com/DrCanD/dikmen-spiking-neurons.git"
    print("[SETUP] installing dikmen-spiking-neurons from GitHub ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", url])
    if r.returncode != 0:                      # PEP 668 externally-managed env
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", url], check=True)
    _clear_dikmen()

from dikmen_neurons import NeuronRegistry   # 39 models / 8 families; we use MS-IF (ISAC)


# ════════════════════════════════════════════════════════════════
# Dataset registry + DIAT folder loader
# ════════════════════════════════════════════════════════════════
REGISTRY_PATH = RESEARCH / 'datasets.json'

def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def register_dataset(name, path, **meta):
    reg = load_registry()
    reg[name] = {'path': str(path), 'registered': datetime.now().isoformat(), **meta}
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(reg, f, indent=2)

def get_dataset_path(name):
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"Dataset '{name}' not in registry {list(reg.keys())}. Register it.")
    p = Path(reg[name]['path'])
    if not p.exists():
        raise FileNotFoundError(f"'{name}' registered at {p} but not found.")
    return p

# register DIAT-uSAT once if missing (folder with X.npy + y.npy)
if 'diat' not in load_registry():
    register_dataset('diat',
        path='/content/drive/MyDrive/NISAC/data/DIAT_uSAT/processed',
        source='DIAT micro-Doppler uSAT', format='folder-npy', classes=6, T=64, features=64,
        notes='X.npy [N,64,64] + y.npy; spectrogram-as-sequence; 80/20 stratified split seed=0')

def load_diat():
    base = get_dataset_path('diat')                       # folder containing X.npy + y.npy
    X = np.load(base / 'X.npy').astype(np.float32)         # [N, 64, 64] = [N, time, Doppler]
    y = np.load(base / 'y.npy').astype(np.int64)
    assert X.ndim == 3, f"expected [N,64,64], got {X.shape}"
    def strat_split(Xa, ya, frac, seed):
        try:
            from sklearn.model_selection import train_test_split
            A, B, ya2, yb2 = train_test_split(Xa, ya, test_size=frac, stratify=ya, random_state=seed)
            return A, ya2, B, yb2
        except Exception:
            rng = np.random.default_rng(seed); ia, ib = [], []
            for c in np.unique(ya):
                ci = np.where(ya == c)[0]; rng.shuffle(ci); k = int((1 - frac) * len(ci))
                ia += list(ci[:k]); ib += list(ci[k:])
            return Xa[ia], ya[ia], Xa[ib], ya[ib]
    Xtv, ytv, Xte, yte = strat_split(X, y, 0.20, 0)                    # 80% trainval / 20% test
    Xtr, ytr, Xva, yva = strat_split(Xtv, ytv, cfg.val_frac / 0.80, 0) # carve val from trainval
    mu, sd = float(Xtr.mean()), float(Xtr.std() + 1e-6)               # standardize with TRAIN stats only
    norm = lambda a: torch.as_tensor((a - mu) / sd).float()
    yy = lambda a: torch.as_tensor(a).long()
    return norm(Xtr), yy(ytr), norm(Xva), yy(yva), norm(Xte), yy(yte)


# ════════════════════════════════════════════════════════════════
# Deep bodies (inline) — sandbox-verified
# ════════════════════════════════════════════════════════════════
def _make_norm(kind, width):
    if kind == "layer":
        return nn.LayerNorm(width)
    if kind == "batch":
        return nn.BatchNorm1d(width)
    return nn.Identity()           # PURE SNN default

class FeatureStack(nn.Module):
    """L (Linear -> [norm] -> neuron) layers; returns (time-avg feat [B,W], last spk [B,T,W])."""
    def __init__(self, in_features, layer_types, width, neuron_kwargs=None, norm="none"):
        super().__init__()
        neuron_kwargs = neuron_kwargs or {}
        dims = [in_features] + [width] * len(layer_types)
        self.projs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(layer_types))])
        self.norms = nn.ModuleList([_make_norm(norm, width) for _ in layer_types])
        self.neurons = nn.ModuleList([NeuronRegistry.create(t, size=width, **neuron_kwargs.get(t, {}))
                                      for t in layer_types])
        self.width = width
    def forward(self, x):
        h = x
        for proj, norm, neuron in zip(self.projs, self.norms, self.neurons):
            B, T, d = h.shape
            z = norm(proj(h.reshape(B*T, d))).reshape(B, T, -1)
            spk, _ = neuron(z)
            h = spk
        return h.mean(dim=1), h
    def layer_firing(self, x):
        h, rates = x, []
        for proj, norm, neuron in zip(self.projs, self.norms, self.neurons):
            B, T, d = h.shape
            z = norm(proj(h.reshape(B*T, d))).reshape(B, T, -1)
            spk, _ = neuron(z); rates.append(spk.float().mean().item()); h = spk
        return rates

class VerticalNet(nn.Module):
    def __init__(self, in_features, n_classes, layer_types, width, neuron_kwargs=None, norm="none"):
        super().__init__()
        self.stack = FeatureStack(in_features, layer_types, width, neuron_kwargs, norm)
        self.readout = nn.Linear(width, n_classes)
    def forward(self, x):
        feat, _ = self.stack(x); return self.readout(feat)
    def firing_rates(self, x):
        return self.stack.layer_firing(x)

class PathBankNet(nn.Module):
    def __init__(self, in_features, n_classes, path_types, width, n_layers,
                 fusion="concat", neuron_kwargs=None, norm="none", shared_stem=True):
        super().__init__()
        if shared_stem:
            self.stem = nn.Linear(in_features, width); path_in = width
        else:
            self.stem = None; path_in = in_features
        self.paths = nn.ModuleList([FeatureStack(path_in, [t]*n_layers, width, neuron_kwargs, norm)
                                    for t in path_types])
        self.readout = nn.Linear(width * len(path_types), n_classes)
        self.n_paths = len(path_types); self.width = width
    def forward(self, x):
        if self.stem is not None:
            B, T, d = x.shape; x = self.stem(x.reshape(B*T, d)).reshape(B, T, -1)
        feats = [p(x)[0] for p in self.paths]
        return self.readout(torch.cat(feats, dim=-1))
    def path_firing(self, x):
        if self.stem is not None:
            B, T, d = x.shape; x = self.stem(x.reshape(B*T, d)).reshape(B, T, -1)
        return [p(x)[1].float().mean().item() for p in self.paths]

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def _solve_width(builder, target, wmax=400):
    best = None
    for w in range(2, wmax):
        p = count_params(builder(w)); d = abs(p - target)
        if best is None or d < best[2]:
            best = (w, p, d)
    return best

@torch.no_grad()
def _calibrate_stack(stack, h, target, iters=30):
    for proj, norm, neuron in zip(stack.projs, stack.norms, stack.neurons):
        B, T, d = h.shape
        z = norm(proj(h.reshape(B*T, d))).reshape(B, T, -1)
        a, b = 1e-3, 1e5
        for _ in range(iters):
            mid = (a*b) ** 0.5; neuron.threshold = mid
            r = neuron(z)[0].float().mean().item()
            a, b = (mid, b) if r > target else (a, mid)
        neuron.threshold = (a*b) ** 0.5
        h = neuron(z)[0]
    return stack

@torch.no_grad()
def calibrate_thresholds(model, x, target):
    if isinstance(model, VerticalNet):
        _calibrate_stack(model.stack, x, target)
    else:
        h = x
        if model.stem is not None:
            B, T, d = x.shape; h = model.stem(x.reshape(B*T, d)).reshape(B, T, -1)
        for p in model.paths:
            _calibrate_stack(p, h, target)
    return model


# ════════════════════════════════════════════════════════════════
# Model builder (capacity widths resolved at runtime against D1)
# ════════════════════════════════════════════════════════════════
def build_model(method, Fin, C, NK, matched):
    n = cfg.norm
    if method == "D0_doppler":
        return VerticalNet(Fin, C, ["Doppler-LIF"]*cfg.L, cfg.W, NK, n)
    if method == "D0_stft":
        return VerticalNet(Fin, C, ["STFT-IF"]*cfg.L, cfg.W, NK, n)
    if method == "D1_H1":
        return VerticalNet(Fin, C, H1_ORDER, cfg.W, NK, n)
    if method == "D1_rev":
        return VerticalNet(Fin, C, H1_REV, cfg.W, NK, n)
    if method == "D1_perm1":
        return VerticalNet(Fin, C, D1_PERM1, cfg.W, NK, n)
    if method == "D0_chirp":
        return VerticalNet(Fin, C, ["Chirp-LIF"]*cfg.L, cfg.W, NK, n)
    if method == "D0_dualtau":
        return VerticalNet(Fin, C, ["Dual-tau-LIF"]*cfg.L, cfg.W, NK, n)
    if method == "D2_concat":
        return PathBankNet(Fin, C, D2_PATHS, matched["d2_w"], cfg.L, "concat", NK, n, True)
    raise KeyError(method)


# ════════════════════════════════════════════════════════════════
# Train / eval / bootstrap
# ════════════════════════════════════════════════════════════════
def save_ckpt(model, opt, epoch, metrics, path):
    torch.save({'epoch': epoch, 'model_state_dict': copy.deepcopy(model.state_dict()),
                'optimizer_state_dict': opt.state_dict(), 'metrics': metrics}, path)

def load_ckpt(model, opt, path):
    if not path.exists():
        return 0, {}
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict']); opt.load_state_dict(ck['optimizer_state_dict'])
    print(f"    [RESUME] from epoch {ck['epoch']}")
    return ck['epoch'] + 1, ck['metrics']

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu()); trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def accuracy(preds, trues, C=None):
    return float((preds == trues).mean())

def macro_f1(preds, trues, C):
    f1s = []
    for c in range(C):
        tp = int(np.sum((preds == c) & (trues == c)))
        fp = int(np.sum((preds == c) & (trues != c)))
        fn = int(np.sum((preds != c) & (trues == c)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)), [round(float(x), 4) for x in f1s]

METRICS = {"accuracy": lambda p, t, C: accuracy(p, t),
           "macro_f1": lambda p, t, C: macro_f1(p, t, C)[0]}

def firing_snapshot(model, xb):
    return model.firing_rates(xb) if isinstance(model, VerticalNet) else model.path_firing(xb)

def train_one(method, seed, Fin, C, NK, matched, tr_loader, va_loader, te_loader, fire_batch, ckpt_dir):
    """Early stopping on SMOOTHED validation (moving avg, denoised); report TEST once at the
    best-smoothed-val epoch (selection on val only -> no test peeking)."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(method, Fin, C, NK, matched).to(device)
    calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)   # init only (config A)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ckpt = ckpt_dir / f'{method}_seed{seed}.pt'
    start, m = load_ckpt(model, opt, ckpt)
    sel = METRICS[cfg.select_metric]
    fire_log     = m.get('fire_log', [])
    val_hist     = m.get('val_hist', [])     # raw per-epoch val scores (for the moving average)
    best_val     = m.get('best_val', -1.0)   # best SMOOTHED val
    best_epoch   = m.get('best_epoch', -1)
    patience_ctr = m.get('patience_ctr', 0)
    best = m.get('best', None)   # {'acc','f1','f1_per_class','preds','trues'} at best-smoothed-val epoch
    for ep in range(start, cfg.max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        vp, vt = evaluate(model, va_loader); v_raw = sel(vp, vt, C)
        val_hist.append(v_raw)
        v_smooth = float(np.mean(val_hist[-cfg.val_smooth:]))   # denoised selection signal
        tp, tt = evaluate(model, te_loader)
        t_acc = accuracy(tp, tt); t_f1, t_f1c = macro_f1(tp, tt, C)
        fr = firing_snapshot(model, fire_batch.to(device))
        improved = v_smooth > best_val + 1e-4
        if improved:                                            # selection on smoothed validation only
            best_val, best_epoch, patience_ctr = v_smooth, ep, 0
            best = {'acc': t_acc, 'f1': t_f1, 'f1_per_class': t_f1c,
                    'preds': tp.tolist(), 'trues': tt.tolist()}  # test captured AT best-smoothed-val epoch
        else:
            patience_ctr += 1
        fire_log.append({'epoch': ep, f'val_{cfg.select_metric}_smooth': round(v_smooth, 4),
                         f'val_{cfg.select_metric}_raw': round(v_raw, 4),
                         'test_acc': round(t_acc, 4), 'test_f1': round(t_f1, 4),
                         'firing': [round(r, 4) for r in fr]})
        save_ckpt(model, opt, ep, {'fire_log': fire_log, 'val_hist': val_hist, 'best_val': best_val,
                                   'best_epoch': best_epoch, 'patience_ctr': patience_ctr,
                                   'best': best}, ckpt)
        star = " *" if improved else ""
        print(f"    ep {ep:>3} val={v_smooth:.4f}(raw {v_raw:.4f}) test_acc={t_acc:.4f} test_f1={t_f1:.4f}{star}")
        if patience_ctr >= cfg.patience:
            print(f"    [EARLY STOP] no smoothed-val gain {cfg.patience} ep; best val={best_val:.4f} @ep{best_epoch} "
                  f"-> test acc={best['acc']:.4f} f1={best['f1']:.4f}")
            break
    return best, fire_log

def paired_bootstrap(preds_a, preds_b, trues, metric_fn, C, nboot=2000, seed=0):
    """Paired bootstrap over test examples; recomputes the metric on each resample."""
    rng = np.random.default_rng(seed); n = len(trues)
    base = metric_fn(preds_a, trues, C) - metric_fn(preds_b, trues, C)
    boots = np.empty(nboot)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = metric_fn(preds_a[i], trues[i], C) - metric_fn(preds_b[i], trues[i], C)
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════
def main():
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M')
    RUN_DIR = PROJECT / 'runs' / RUN_ID
    CKPT_DIR = RUN_DIR / 'checkpoints'
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── banner + manifest ──
    print(f"[PERSIST] project: {PROJECT}")
    print(f"[HW] device={device}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"[PROGRESS] Experiment {cfg.exp_index}/{cfg.exp_total}: {cfg.exp_name}")
    for idx, name, status in cfg.exp_manifest:
        mark = "->" if status == "THIS" else ("x" if status == "done" else "  ")
        print(f"   {mark} {idx}/{cfg.exp_total}: {name} [{status}]")

    # ── data (train / val / test; val is for early-stopping model selection only) ──
    Xtr, ytr, Xva, yva, Xte, yte = load_diat()
    Fin, C = Xtr.shape[-1], int(max(ytr.max(), yva.max(), yte.max()) + 1)
    print(f"[DATA] DIAT train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)} F={Fin} C={C}")
    print(f"[DATA] standardized range [{Xtr.min():.3f}, {Xtr.max():.3f}]  (chance={1/C:.3f})")
    NK = {"STFT-IF": {"window_len": cfg.stft_window}}
    fire_batch = Xtr[:cfg.batch_size]
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True,
                           num_workers=2, pin_memory=(device.type == 'cuda'), persistent_workers=True)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

    # ── capacity matching against D1 (homogeneous D0_* at W=128 are already ~matched: neuron params are O(W)) ──
    d1_params = count_params(VerticalNet(Fin, C, H1_ORDER, cfg.W, NK, cfg.norm))
    d2_w, d2_p, _ = _solve_width(
        lambda w: PathBankNet(Fin, C, D2_PATHS, w, cfg.L, "concat", NK, cfg.norm, True), d1_params)
    matched = {"d2_w": d2_w}
    d0_dop_p = count_params(VerticalNet(Fin, C, ["Doppler-LIF"]*cfg.L, cfg.W, NK, cfg.norm))
    print(f"[CAPACITY] D1 params={d1_params}  (homog D0 at W={cfg.W} ~ {d0_dop_p}, {100*abs(d0_dop_p-d1_params)/d1_params:.2f}%)")
    print(f"[CAPACITY] D2 concat width={d2_w} -> {d2_p} ({100*abs(d2_p-d1_params)/d1_params:.2f}%)")

    # ── PRE-REGISTERED predictions (printed before any result is seen) ──
    print("[PRE-REGISTERED] directional bets, fixed before results:")
    print("   P1: D2 (horizontal) > best homogeneous, sign consistent across seeds")
    print("   P2: D2 > D1 (horizontal beats vertical)")
    print("   P3: D1(H1) > reverse and permutation (layer order matters)")
    print("   P4: D1 vs best-homog is LESS negative than on SHD (STFT is in-domain on radar)")
    print("   P5: best homogeneous is a spectro-temporal resonator (Doppler/Chirp/STFT), not worst")

    # ── workload ──
    n_runs = len(cfg.methods) * len(cfg.seeds)
    print(f"[WORKLOAD] {len(cfg.methods)} methods x {len(cfg.seeds)} seeds = {n_runs} runs; "
          f"early stop on val (patience {cfg.patience}, max {cfg.max_epochs} ep)")

    # ── run ──
    results = {}; run_counter = 0; first_t = None
    for method in cfg.methods:
        accs, f1s, preds, trues, f1cs = [], [], [], [], []
        for seed in cfg.seeds:
            run_counter += 1
            print(f"\n  Run {run_counter}/{n_runs}: {METHOD_LABELS[method]} | seed={seed}")
            t0 = time.time()
            best, flog = train_one(method, seed, Fin, C, NK, matched,
                                   tr_loader, va_loader, te_loader, fire_batch, CKPT_DIR)
            dt = time.time() - t0
            if first_t is None:
                first_t = dt; print(f"    [ETA] ~{first_t*(n_runs-1)/60:.0f} min for remaining {n_runs-1} runs (early stop varies)")
            accs.append(best['acc']); f1s.append(best['f1']); f1cs.append(best['f1_per_class'])
            preds.append(best['preds']); trues.append(best['trues'])
            with open(RUN_DIR / f'results_{method}_seed{seed}.json', 'w') as f:
                json.dump({'method': method, 'seed': seed, 'acc': best['acc'], 'f1': best['f1'],
                           'f1_per_class': best['f1_per_class'], 'preds': best['preds'],
                           'trues': best['trues'], 'fire_log': flog, 'time_s': dt}, f, indent=2)
        results[method] = {'accs': accs, 'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
                           'f1s': f1s, 'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                           'f1_per_class': f1cs, 'preds': preds, 'trues': trues}
        print(f"  => {method}: acc {np.mean(accs):.4f}+/-{np.std(accs):.4f}  "
              f"macroF1 {np.mean(f1s):.4f}+/-{np.std(f1s):.4f}  per-class-F1[seed0]={f1cs[0]}")

    # ── paired bootstrap: control = capacity-matched BEST homogeneous by PRIMARY metric (most stringent) ──
    primary = 'f1_mean' if cfg.select_metric == 'macro_f1' else 'acc_mean'
    homog = [m for m in results if m.startswith("D0_")]
    best_homog = max(homog, key=lambda m: results[m][primary]) if homog else None
    if best_homog:
        print(f"\n[BOOTSTRAP] control = best homogeneous = {best_homog} "
              f"(acc {results[best_homog]['acc_mean']:.4f}, F1 {results[best_homog]['f1_mean']:.4f}); paired on test, both metrics")
    else:
        print("\n[BOOTSTRAP] no homogeneous baseline present")

    def boot_pair(a, b):
        out = {}
        for label, mfn in [("F1", METRICS["macro_f1"]), ("acc", METRICS["accuracy"])]:
            per_seed = [paired_bootstrap(np.array(results[a]['preds'][si]), np.array(results[b]['preds'][si]),
                                         np.array(results[a]['trues'][si]), mfn, C, seed=cfg.seeds[si])
                        for si in range(len(cfg.seeds))]
            diffs = [d for d, _, _ in per_seed]
            sig = all(lo > 0 for _, lo, _ in per_seed) or all(hi < 0 for _, _, hi in per_seed)
            print(f"   {a} - {b} [{label}]: mean {np.mean(diffs):+.4f}  "
                  f"CIs {[(round(l,3), round(h,3)) for _, l, h in per_seed]}  consistent={sig}")
            out[label] = per_seed
        return out

    boot = {}
    if best_homog:
        for m in ["D1_H1", "D2_concat", "D1_rev"]:
            if m in results:
                boot[f"{m}_vs_bestHomog({best_homog})"] = boot_pair(m, best_homog)
    if "D2_concat" in results and "D1_H1" in results:
        boot["horizontal_vs_vertical"] = boot_pair("D2_concat", "D1_H1")     # the core question
    if "D1_H1" in results and "D1_rev" in results:
        boot["ordering_H1_vs_rev"] = boot_pair("D1_H1", "D1_rev")            # does layer order matter
    if "D1_H1" in results and "D1_perm1" in results:
        boot["ordering_H1_vs_perm1"] = boot_pair("D1_H1", "D1_perm1")

    # ── summary ──
    summary = {'run_id': RUN_ID, 'exp_index': cfg.exp_index, 'select_metric': cfg.select_metric,
               'config': {k: v for k, v in vars(cfg).items()},
               'capacity': {'d1_params': d1_params, 'd2_w': d2_w}, 'best_homog': best_homog,
               'results': {m: {'acc_mean': r['acc_mean'], 'acc_std': r['acc_std'], 'accs': r['accs'],
                               'f1_mean': r['f1_mean'], 'f1_std': r['f1_std'], 'f1s': r['f1s'],
                               'f1_per_class': r['f1_per_class']} for m, r in results.items()},
               'bootstrap': boot, 'timestamp': datetime.now().isoformat()}
    for name in ['summary.json', 'latest_summary.json']:
        with open((RUN_DIR if name == 'summary.json' else PROJECT) / name, 'w') as f:
            json.dump(summary, f, indent=2)

    # ── firing-stability plot (pure-SNN drift check) ──
    try:
        plt.figure(figsize=(7, 4))
        flog = json.load(open(RUN_DIR / f'results_D1_H1_seed{cfg.seeds[0]}.json'))['fire_log']
        arr = np.array([e['firing'] for e in flog])
        for li in range(arr.shape[1]):
            plt.plot(range(len(arr)), arr[:, li], marker='o', label=f'layer {li} ({H1_ORDER[li]})')
        plt.axhspan(cfg.fire_lo, cfg.fire_hi, color='green', alpha=0.06, label='healthy band')
        plt.xlabel('epoch'); plt.ylabel('firing rate'); plt.title('D1(H1) per-layer firing (config A: init-calibrated, free during training)')
        plt.legend(fontsize=7); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[plot skipped] {e}")

    print(f"\n[DONE] summary -> {RUN_DIR / 'summary.json'}")
    return summary

summary = main()
