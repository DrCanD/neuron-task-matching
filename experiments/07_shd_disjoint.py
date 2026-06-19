# ============================================================================
# A cross-domain double dissociation of neuron-task matching in pure
# spiking networks  --  Dikmen & Karadag
#
# Verified, locked experiment script. The body below is the exact code that
# produced the paper numbers; it is kept self-contained for Colab paste-and-run.
# Requires the dikmen-spiking-neurons library (NeuronRegistry). See README.
#
# SHD official speaker-disjoint re-run (primary audio result)
# Paper: Section 3.3 split definition, Table 3 (disjoint ranking, Doppler 6th) and
# Figure 4 (Spearman rho = 0.76 between mixed and disjoint orderings). Self-downloads
# SHD from zenkelab.org and bins to 12 frames; no Drive cache required.
# Source cell: "Shd deep probe" notebook, cell 6.
# ============================================================================

# ═══════════════════════════════════════════════════════════════════════════
# SHD SPEAKER-DISJOINT  ·  per-neuron ranking re-run  (reviewer fix #2)
# ───────────────────────────────────────────────────────────────────────────
# WHY: the paper's audio ranking (Table 4) and the double dissociation (Table 5)
# were measured on a speaker-MIXED re-split cache. A reviewer will ask whether
# the ranking survives the OFFICIAL speaker-DISJOINT split. This cell changes
# ONLY the split (official zenkelab.org SHD train/test, disjoint by construction:
# 2 of 12 speakers appear only in test) and keeps EVERYTHING else identical to
# the ablation/dissociation harness: L=4, W=128, pure-SNN (no norm), init-only
# threshold calibration to target firing, Adam+CE, temporal-mean rate readout,
# val-smoothed-macro-F1 selection (mean of last 5 epochs) + patience early stop,
# 3 seeds, paired bootstrap (2000 resamples, "consistent" = same sign all seeds).
#
# PRE-REGISTERED PREDICTION (fixed before looking at the numbers):
#   - Absolute macro-F1 DROPS on the disjoint split (it is the harder, correct split).
#   - The neuron RANKING is preserved: CrossInhib-LIF stays at/near the top and
#     Doppler-LIF stays near the bottom on audio.
#   - Headline statistics: (a) Spearman rank correlation between the locked
#     mixed-split ranking (Table 4) and this disjoint ranking; (b) the audio-side
#     CrossInhib-LIF vs Doppler-LIF paired bootstrap, expected consistent in sign.
#   A high Spearman + a consistent CrossInhib>Doppler on audio confirms the
#   dissociation is split-invariant. A scrambled ranking would falsify it.
#
# Single paste-and-run Colab cell. Checkpoints to Drive → resumes after a drop.
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, json, gzip, shutil, time, copy, subprocess, warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
warnings.filterwarnings("ignore")

try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False


# ═══════════════════════════════════════════════════════════════════════════
# Config (no argparse) + experiment manifest
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    # architecture (LOCKED — identical to the ablation/dissociation harness)
    L: int = 4
    W: int = 128
    norm: str = "none"                 # PURE SNN: no normalization anywhere
    target_firing: float = 0.15        # init threshold calibration target
    stft_window: int = 8               # SHD T=12 -> STFT window must be <= T

    # SHD binning
    T: int = 12                        # frames (matches the paper's cached SHD)
    F: int = 700                       # cochlear channels
    duration_s: float = 1.0            # bin spike times over [0, duration] into T frames

    # training / selection (LOCKED)
    max_epochs: int = 200
    patience: int = 20                 # early stop on smoothed-val plateau
    val_smooth: int = 5                # mean over last 5 epochs (paper's wording)
    val_frac: float = 0.15             # val carved from official train (selection ONLY)
    select_metric: str = "macro_f1"
    lr: float = 1e-3
    batch_size: int = 128
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    nboot: int = 2000

    # the 8 neurons of Table 4 (Vanilla-LIF registered below; rest are MS-IF)
    methods: list = field(default_factory=lambda: [
        "CrossInhib-LIF", "Vanilla-LIF", "Dual-tau-LIF", "Chirp-LIF",
        "Beam-IF", "Doppler-LIF", "STFT-IF", "Phase-LIF"])

    # locked speaker-MIXED ranking (Table 4) — reference for the Spearman check.
    # (values are the paper's reported macro-F1; used ONLY for rank correlation)
    mixed_ranking: dict = field(default_factory=lambda: {
        "CrossInhib-LIF": 0.741, "Vanilla-LIF": 0.703, "Dual-tau-LIF": 0.694,
        "Chirp-LIF": 0.662, "Beam-IF": 0.639, "Doppler-LIF": 0.601,
        "STFT-IF": 0.244, "Phase-LIF": 0.143})

    # experiment manifest (single source of truth)
    exp_index: int = 2
    exp_total: int = 3
    exp_name: str = "SHD speaker-disjoint per-neuron ranking"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "DIAT ablation + dissociation (mixed-split) [paper]", "done"),
        (2, "SHD speaker-DISJOINT ranking re-run", "THIS"),
        (3, "parameter-matched CNN baseline", "pending"),
    ])

cfg = Config()

# fast sandbox check:  REDUCED=1 python shd_disjoint_ranking.py
if os.environ.get("REDUCED") == "1":
    cfg.L = 2; cfg.W = 24
    cfg.max_epochs = 3; cfg.patience = 2; cfg.val_smooth = 2
    cfg.seeds = [42]; cfg.nboot = 200
    cfg.methods = ["CrossInhib-LIF", "Doppler-LIF", "STFT-IF", "Vanilla-LIF"]


# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    RESEARCH = Path("/content/drive/MyDrive/Research")
else:
    RESEARCH = Path("/home/claude/research")
RESEARCH.mkdir(parents=True, exist_ok=True)
PROJECT = RESEARCH / "NISAC_DeepHetero"
PROJECT.mkdir(parents=True, exist_ok=True)
CACHE = PROJECT / "shd_official"
CACHE.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# dikmen-spiking-neurons import (mandatory sys.modules cleanup + GitHub fallback)
# ═══════════════════════════════════════════════════════════════════════════
def _clear_dikmen():
    for k in [k for k in sys.modules if k.startswith("dikmen")]:
        del sys.modules[k]

_clear_dikmen()
try:
    import dikmen_neurons  # noqa
except ImportError:
    url = "git+https://github.com/DrCanD/dikmen-spiking-neurons.git"
    print("[SETUP] installing dikmen-spiking-neurons from GitHub ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", url])
    if r.returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--break-system-packages", url], check=True)
    _clear_dikmen()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt   # Colab handles backend; never matplotlib.use('Agg')
import dikmen_neurons as D
from dikmen_neurons import BaseNeuron, spike_hard, NeuronRegistry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ═══════════════════════════════════════════════════════════════════════════
# Matched Vanilla-LIF baseline  (same as the HPO/ablation: zero learnable neuron params)
# ═══════════════════════════════════════════════════════════════════════════
class VanillaLIF(BaseNeuron):
    _family = "lif"
    _description = "Standard leaky integrate-and-fire baseline (control)."
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        mem = mem * (1.0 - spk)
        return spk, {"mem": mem}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF


# ═══════════════════════════════════════════════════════════════════════════
# Dataset registry
# ═══════════════════════════════════════════════════════════════════════════
REGISTRY_PATH = RESEARCH / "datasets.json"

def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def register_dataset(name, path, **meta):
    reg = load_registry()
    reg[name] = {"path": str(path), "registered": datetime.now().isoformat(), **meta}
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)
    print(f"[REGISTRY] {name} -> {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Fast download (aria2c > wget > requests)
# ═══════════════════════════════════════════════════════════════════════════
def fast_download(url, dest, desc="file"):
    dest = Path(dest)
    if dest.exists():
        print(f"[CACHE] {desc} already at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not shutil.which("aria2c") and IN_COLAB:
        subprocess.run(["apt-get", "install", "-y", "aria2", "-qq"], capture_output=True)
    if shutil.which("aria2c"):
        subprocess.run(["aria2c", "-x", "16", "-s", "16", "-k", "1M",
                        "--dir", str(dest.parent), "--out", dest.name, url], check=True)
    elif shutil.which("wget"):
        subprocess.run(["wget", "-q", "--show-progress", "-O", str(dest), url], check=True)
    else:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
    print(f"[OK] {desc} -> {dest}")
    return dest


# ═══════════════════════════════════════════════════════════════════════════
# OFFICIAL SHD loader  (speaker-disjoint train/test; bin spikes -> [N, T, 700])
# ═══════════════════════════════════════════════════════════════════════════
# Mirrors: zenkelab.org/datasets and compneuro.net/datasets
SHD_URLS = {
    "train": ["https://zenkelab.org/datasets/shd_train.h5.gz",
              "https://compneuro.net/datasets/shd_train.h5.gz"],
    "test":  ["https://zenkelab.org/datasets/shd_test.h5.gz",
              "https://compneuro.net/datasets/shd_test.h5.gz"],
}

def _download_shd(split):
    gz = CACHE / f"shd_{split}.h5.gz"
    h5 = CACHE / f"shd_{split}.h5"
    if not h5.exists():
        last = None
        for u in SHD_URLS[split]:
            try:
                fast_download(u, gz, desc=f"SHD {split}")
                last = None
                break
            except Exception as e:
                last = e
                print(f"[WARN] {u} failed: {e}")
        if last is not None:
            raise last
        with gzip.open(gz, "rb") as fi, open(h5, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return h5

def _bin_h5(h5_path, T, F, duration):
    """SHD HDF5 -> [N, T, F] spike-count frames. Keys: spikes/times, spikes/units, labels."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        N = len(labels)
        X = np.zeros((N, T, F), dtype=np.float32)
        for i in range(N):
            t = np.asarray(times[i], dtype=np.float32)
            u = np.asarray(units[i], dtype=np.int64)
            if t.size == 0:
                continue
            b = np.clip((t / duration * T).astype(np.int64), 0, T - 1)
            u = np.clip(u, 0, F - 1)
            np.add.at(X, (np.full(b.shape, i), b, u), 1.0)
    return X, labels

def _strat_split(X, y, frac, seed):
    try:
        from sklearn.model_selection import train_test_split
        A, B, ya, yb = train_test_split(X, y, test_size=frac, stratify=y, random_state=seed)
        return A, ya, B, yb
    except Exception:
        rng = np.random.default_rng(seed); ia, ib = [], []
        for c in np.unique(y):
            ci = np.where(y == c)[0]; rng.shuffle(ci); k = int((1 - frac) * len(ci))
            ia += list(ci[:k]); ib += list(ci[k:])
        return X[ia], y[ia], X[ib], y[ib]

def load_shd_disjoint():
    """Returns train/val/test tensors. test = OFFICIAL disjoint test; val carved from train."""
    if IN_COLAB:
        tr_h5, te_h5 = _download_shd("train"), _download_shd("test")
    else:
        # sandbox: synthetic SHD-structured HDF5 (only to verify the pipeline runs)
        tr_h5, te_h5 = _make_synth_h5("train"), _make_synth_h5("test")
    Xtr_full, ytr_full = _bin_h5(tr_h5, cfg.T, cfg.F, cfg.duration_s)
    Xte, yte = _bin_h5(te_h5, cfg.T, cfg.F, cfg.duration_s)
    # carve a stratified validation set from the official TRAIN (selection only)
    Xtr, ytr, Xva, yva = _strat_split(Xtr_full, ytr_full, cfg.val_frac, seed=0)
    register_dataset("shd_official",
        path=str(CACHE), source="zenkelab.org", format="hdf5", split="speaker-disjoint",
        classes=int(max(ytr_full.max(), yte.max()) + 1), T=cfg.T, features=cfg.F,
        notes="official train/test; 2 of 12 speakers test-only; binned to T frames")
    to_t = lambda a: torch.as_tensor(a).float()
    to_y = lambda a: torch.as_tensor(a).long()
    return (to_t(Xtr), to_y(ytr), to_t(Xva), to_y(yva), to_t(Xte), to_y(yte))

def _make_synth_h5(split):
    """Tiny synthetic file with the real SHD layout, for the local REDUCED check only."""
    import h5py
    p = CACHE / f"synth_{split}.h5"
    if p.exists():
        return p
    rng = np.random.default_rng(0 if split == "train" else 1)
    N, C = (240 if split == "train" else 80), 20
    y = rng.integers(0, C, N).astype(np.int64)
    dt_t = h5py.special_dtype(vlen=np.dtype("float32"))
    dt_u = h5py.special_dtype(vlen=np.dtype("int64"))
    with h5py.File(p, "w") as f:
        g = f.create_group("spikes")
        ts = g.create_dataset("times", (N,), dtype=dt_t)
        us = g.create_dataset("units", (N,), dtype=dt_u)
        for i in range(N):
            k = rng.integers(40, 120)
            # class-correlated channel band so a classifier can actually learn
            lo = (y[i] * 700 // C)
            ts[i] = np.sort(rng.uniform(0, 1.0, k)).astype(np.float32)
            us[i] = np.clip(rng.normal(lo, 60, k), 0, 699).astype(np.int64)
        f.create_dataset("labels", data=y)
    return p


# ═══════════════════════════════════════════════════════════════════════════
# Deep bodies (inline) — verbatim from the ablation/dissociation harness (norm="none")
# ═══════════════════════════════════════════════════════════════════════════
def _make_norm(kind, width):
    if kind == "layer": return nn.LayerNorm(width)
    if kind == "batch": return nn.BatchNorm1d(width)
    return nn.Identity()                       # PURE SNN default

class FeatureStack(nn.Module):
    def __init__(self, in_features, layer_types, width, neuron_kwargs=None, norm="none"):
        super().__init__()
        neuron_kwargs = neuron_kwargs or {}
        dims = [in_features] + [width] * len(layer_types)
        self.projs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(layer_types))])
        self.norms = nn.ModuleList([_make_norm(norm, width) for _ in layer_types])
        self.neurons = nn.ModuleList(
            [NeuronRegistry.create(t, size=width, **neuron_kwargs.get(t, {})) for t in layer_types])
        self.width = width
    def forward(self, x):
        h = x
        for proj, norm, neuron in zip(self.projs, self.norms, self.neurons):
            B, T, d = h.shape
            z = norm(proj(h.reshape(B*T, d))).reshape(B, T, -1)
            spk, _ = neuron(z); h = spk
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
        self.layer_types = list(layer_types)
    def forward(self, x):
        feat, _ = self.stack(x); return self.readout(feat)
    def firing_rates(self, x):
        return self.stack.layer_firing(x)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def _calibrate_stack(stack, h, target, iters=30):
    """Bisection: set each layer's threshold so init firing ~= target. The spike
    equation (mem >= threshold) is unchanged; threshold is a per-layer scalar."""
    for proj, norm, neuron in zip(stack.projs, stack.norms, stack.neurons):
        B, T, d = h.shape
        z = norm(proj(h.reshape(B*T, d))).reshape(B, T, -1)
        a, b = 1e-3, 1e5
        for _ in range(iters):
            mid = (a*b)**0.5; neuron.threshold = mid
            r = neuron(z)[0].float().mean().item()
            if r > target: a = mid
            else:          b = mid
        neuron.threshold = (a*b)**0.5
        h = neuron(z)[0]
    return stack

@torch.no_grad()
def calibrate_thresholds(model, x, target):
    _calibrate_stack(model.stack, x, target)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Metrics / eval / bootstrap  (verbatim from the ablation)
# ═══════════════════════════════════════════════════════════════════════════
def accuracy(p, t, C=None): return float((p == t).mean())

def macro_f1(p, t, C):
    f1s = []
    for c in range(C):
        tp = int(np.sum((p == c) & (t == c))); fp = int(np.sum((p == c) & (t != c)))
        fn = int(np.sum((p != c) & (t == c)))
        pr = tp/(tp+fp) if (tp+fp) else 0.0; rc = tp/(tp+fn) if (tp+fn) else 0.0
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) else 0.0)
    return float(np.mean(f1s)), [round(float(x), 4) for x in f1s]

METRICS = {"accuracy": lambda p, t, C: accuracy(p, t),
           "macro_f1": lambda p, t, C: macro_f1(p, t, C)[0]}

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).argmax(1).cpu()); trues.append(yb)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def firing_snapshot(model, xb):
    return model.firing_rates(xb)

def paired_bootstrap(preds_a, preds_b, trues, metric_fn, C, nboot=2000, seed=0):
    """Paired bootstrap over test examples; resamples preds AND trues by the same indices."""
    rng = np.random.default_rng(seed); n = len(trues)
    base = metric_fn(preds_a, trues, C) - metric_fn(preds_b, trues, C)
    boots = np.empty(nboot)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = metric_fn(preds_a[i], trues[i], C) - metric_fn(preds_b[i], trues[i], C)
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ═══════════════════════════════════════════════════════════════════════════
# Model builder (8 homogeneous single-neuron nets) + train_one (verbatim)
# ═══════════════════════════════════════════════════════════════════════════
def build_model(method, Fin, C, NK):
    return VerticalNet(Fin, C, [method] * cfg.L, cfg.W, NK, cfg.norm)

def save_ckpt(model, opt, epoch, metrics, path):
    torch.save({"epoch": epoch, "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": opt.state_dict(), "metrics": metrics}, path)

def load_ckpt(model, opt, path):
    if not path.exists():
        return 0, {}
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer_state_dict"])
    print(f"    [RESUME] from epoch {ck['epoch']}")
    return ck["epoch"] + 1, ck["metrics"]

def train_one(method, seed, Fin, C, NK, tr_loader, va_loader, te_loader, fire_batch, ckpt_dir):
    """Early stop on SMOOTHED validation; report TEST once at best-smoothed-val epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(method, Fin, C, NK).to(device)
    calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)   # init only
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ckpt = ckpt_dir / f"{method}_seed{seed}.pt"
    start, m = load_ckpt(model, opt, ckpt)
    sel = METRICS[cfg.select_metric]
    fire_log     = m.get("fire_log", [])
    val_hist     = m.get("val_hist", [])
    best_val     = m.get("best_val", -1.0)
    best_epoch   = m.get("best_epoch", -1)
    patience_ctr = m.get("patience_ctr", 0)
    best = m.get("best", None)
    for ep in range(start, cfg.max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        vp, vt = evaluate(model, va_loader); v_raw = sel(vp, vt, C)
        val_hist.append(v_raw)
        v_smooth = float(np.mean(val_hist[-cfg.val_smooth:]))
        tp, tt = evaluate(model, te_loader)
        t_acc = accuracy(tp, tt); t_f1, t_f1c = macro_f1(tp, tt, C)
        fr = firing_snapshot(model, fire_batch.to(device))
        improved = v_smooth > best_val + 1e-4
        if improved:
            best_val, best_epoch, patience_ctr = v_smooth, ep, 0
            best = {"acc": t_acc, "f1": t_f1, "f1_per_class": t_f1c,
                    "firing": [round(r, 4) for r in fr],
                    "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            patience_ctr += 1
        fire_log.append({"epoch": ep, "val_smooth": round(v_smooth, 4), "val_raw": round(v_raw, 4),
                         "test_acc": round(t_acc, 4), "test_f1": round(t_f1, 4),
                         "firing": [round(r, 4) for r in fr]})
        save_ckpt(model, opt, ep, {"fire_log": fire_log, "val_hist": val_hist, "best_val": best_val,
                                   "best_epoch": best_epoch, "patience_ctr": patience_ctr,
                                   "best": best}, ckpt)
        star = " *" if improved else ""
        print(f"    ep {ep:>3} val={v_smooth:.4f}(raw {v_raw:.4f}) test_acc={t_acc:.4f} test_f1={t_f1:.4f}{star}")
        if patience_ctr >= cfg.patience:
            print(f"    [EARLY STOP] no smoothed-val gain {cfg.patience} ep; best @ep{best_epoch} "
                  f"-> test f1={best['f1']:.4f}")
            break
    return best, fire_log


# ═══════════════════════════════════════════════════════════════════════════
# Rank correlation (Spearman, manual: Pearson on ranks; no scipy dependency)
# ═══════════════════════════════════════════════════════════════════════════
def spearman(names, score_x, score_y):
    xs = np.array([score_x[n] for n in names], dtype=float)
    ys = np.array([score_y[n] for n in names], dtype=float)
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    RUN_ID = os.environ.get("RUN_ID", "main")   # STABLE across sessions -> resumes after a Colab drop; set RUN_ID=... or delete the dir to start fresh
    RUN_DIR = PROJECT / "runs_shd_disjoint" / RUN_ID
    CKPT_DIR = RUN_DIR / "checkpoints"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── banner + manifest ──
    print(f"[PERSIST] project: {PROJECT}")
    print(f"[HW] device={device}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"[PROGRESS] Experiment {cfg.exp_index}/{cfg.exp_total}: {cfg.exp_name}")
    for idx, name, status in cfg.exp_manifest:
        mark = "->" if status == "THIS" else ("x" if status == "done" else "  ")
        print(f"   {mark} {idx}/{cfg.exp_total}: {name} [{status}]")

    # ── data (official disjoint test; val carved from official train) ──
    Xtr, ytr, Xva, yva, Xte, yte = load_shd_disjoint()
    Fin = Xtr.shape[-1]; C = int(max(ytr.max(), yva.max(), yte.max()) + 1)
    print(f"[DATA] SHD-disjoint train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)}  C={C}")
    print(f"[DATA] spike-count range [{Xtr.min():.1f}, {Xtr.max():.1f}]  chance={1/C:.3f}")
    NK = {"STFT-IF": {"window_len": cfg.stft_window}}
    fire_batch = Xtr[:cfg.batch_size]
    pin = (device.type == "cuda")
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True,
                           num_workers=2, pin_memory=pin, persistent_workers=pin)
    va_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

    # ── workload ──
    n_runs = len(cfg.methods) * len(cfg.seeds)
    print(f"[WORKLOAD] {len(cfg.methods)} neurons x {len(cfg.seeds)} seeds = {n_runs} runs x <= {cfg.max_epochs} ep")

    # ── run ──
    results = {}; run_counter = 0; first_t = None
    for method in cfg.methods:
        f1s, accs, preds_seeds, trues_seeds = [], [], [], []
        for seed in cfg.seeds:
            run_counter += 1
            print(f"\n  Run {run_counter}/{n_runs}: {method} | seed={seed}")
            t0 = time.time()
            best, flog = train_one(method, seed, Fin, C, NK,
                                   tr_loader, va_loader, te_loader, fire_batch, CKPT_DIR)
            dt = time.time() - t0
            if first_t is None:
                first_t = dt; print(f"    [ETA] ~{first_t*(n_runs-1)/60:.0f} min for remaining {n_runs-1} runs")
            f1s.append(best["f1"]); accs.append(best["acc"])
            preds_seeds.append(best["preds"]); trues_seeds.append(best["trues"])
            with open(RUN_DIR / f"results_{method}_seed{seed}.json", "w") as f:
                json.dump({"method": method, "seed": seed, "f1": best["f1"], "acc": best["acc"],
                           "f1_per_class": best["f1_per_class"], "firing": best["firing"],
                           "fire_log": flog, "time_s": dt}, f, indent=2)
        results[method] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                           "acc_mean": float(np.mean(accs)), "f1s": f1s,
                           "preds": preds_seeds, "trues": trues_seeds}
        print(f"  => {method}: macro-F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # ── disjoint ranking + Spearman vs the locked mixed ranking ──
    disjoint = {m: results[m]["f1_mean"] for m in cfg.methods}
    order_dis = sorted(cfg.methods, key=lambda m: disjoint[m], reverse=True)
    order_mix = sorted(cfg.methods, key=lambda m: cfg.mixed_ranking[m], reverse=True)
    rho = spearman(cfg.methods, cfg.mixed_ranking, disjoint)

    print("\n" + "=" * 74)
    print(" SHD per-neuron ranking: locked MIXED (Table 4)  vs  OFFICIAL DISJOINT")
    print("=" * 74)
    print(f"  {'neuron':<16}{'mixed F1':>10}{'mixed rk':>9}   {'disjoint F1':>12}{'disjoint rk':>12}")
    for m in order_mix:
        print(f"  {m:<16}{cfg.mixed_ranking[m]:>10.3f}{order_mix.index(m)+1:>9}   "
              f"{disjoint[m]:>12.4f}{order_dis.index(m)+1:>12}")
    print(f"\n  Spearman rank correlation (mixed vs disjoint) = {rho:+.3f}")
    print(f"  mixed   order: {' > '.join(order_mix)}")
    print(f"  disjoint order: {' > '.join(order_dis)}")

    # ── audio-side dissociation: CrossInhib-LIF vs Doppler-LIF, paired bootstrap per seed ──
    boot = None
    a, b = "CrossInhib-LIF", "Doppler-LIF"
    if a in results and b in results:
        per_seed = []
        for si in range(len(cfg.seeds)):
            pa = np.array(results[a]["preds"][si]); pb = np.array(results[b]["preds"][si])
            tr = np.array(results[a]["trues"][si])
            per_seed.append(paired_bootstrap(pa, pb, tr, METRICS["macro_f1"], C, nboot=cfg.nboot,
                                             seed=cfg.seeds[si]))
        diffs = [d for d, _, _ in per_seed]
        consistent = all(lo > 0 for _, lo, _ in per_seed) or all(hi < 0 for _, _, hi in per_seed)
        boot = {"pair": f"{a} - {b}", "mean_diff": float(np.mean(diffs)),
                "per_seed_ci": [(round(l, 3), round(h, 3)) for _, l, h in per_seed],
                "consistent_sign": consistent}
        print(f"\n[BOOTSTRAP audio] {a} - {b}: mean macro-F1 diff {np.mean(diffs):+.4f}  "
              f"per-seed CIs {boot['per_seed_ci']}  consistent_sign={consistent}")
        print("  (paper Table 5: CrossInhib beats Doppler on audio by ~0.140, consistent)")

    # ── summary ──
    summary = {"run_id": RUN_ID, "split": "speaker-disjoint (official zenkelab.org)",
               "config": {k: v for k, v in vars(cfg).items()},
               "results": {m: {"f1_mean": results[m]["f1_mean"], "f1_std": results[m]["f1_std"],
                               "acc_mean": results[m]["acc_mean"], "f1s": results[m]["f1s"]}
                           for m in cfg.methods},
               "disjoint_ranking": order_dis, "mixed_ranking": order_mix,
               "spearman_mixed_vs_disjoint": rho, "crossinhib_vs_doppler_audio": boot,
               "timestamp": datetime.now().isoformat()}
    for nm, base in [("summary.json", RUN_DIR), ("latest_shd_disjoint.json", PROJECT)]:
        with open(base / nm, "w") as f:
            json.dump(summary, f, indent=2)

    # ── plot: mixed vs disjoint ranking ──
    try:
        ys_mix = [cfg.mixed_ranking[m] for m in order_dis]
        ys_dis = [disjoint[m] for m in order_dis]
        x = np.arange(len(order_dis))
        plt.figure(figsize=(9, 4))
        plt.plot(x, ys_mix, "o--", label="mixed split (Table 4)")
        plt.plot(x, ys_dis, "s-", label="official disjoint split")
        plt.xticks(x, order_dis, rotation=40, ha="right", fontsize=8)
        plt.ylabel("macro-F1"); plt.title(f"SHD per-neuron ranking  (Spearman rho={rho:+.2f})")
        plt.legend(); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[plot skipped] {e}")

    print(f"\n[DONE] summary -> {RUN_DIR / 'summary.json'}")
    return summary

summary = main()
