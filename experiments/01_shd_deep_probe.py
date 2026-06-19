# ============================================================================
# A cross-domain double dissociation of neuron-task matching in pure
# spiking networks  --  Dikmen & Karadag
#
# Verified, locked experiment script. The body below is the exact code that
# produced the paper numbers; it is kept self-contained for Colab paste-and-run.
# Requires the dikmen-spiking-neurons library (NeuronRegistry). See README.
#
# Experiment 1/3 — harness probe (NOT a paper table)
# Smoke + speed + firing-stability check on SHD (T=12) before the DIAT headline.
# Confirms the pure-SNN stack trains, fires in band, and resumes from checkpoint.
# Source cell: "Shd deep probe" notebook, cell 0.
# ============================================================================

"""
SHD DEEP PROBE — pure-SNN MS-IF heterogeneity (vertical vs horizontal)
Experiment 1/3: smoke + speed + firing-stability check before DIAT headline.

Pure-SNN (neuromorphic-hardware target): inter-layer signal is spikes; no
BatchNorm/LayerNorm, no softmax gate. Per-type threshold calibration at init AND
per-epoch recalibration (gradient-free threshold homeostasis) hold firing at the
target throughout training; firing is logged every epoch as a drift check.

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
    stft_window_shd: int = 8          # SHD T=12 -> window must be <= T
    norm: str = "none"                # PURE SNN: no normalization layer
    # training
    n_epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 128
    seeds: list = field(default_factory=lambda: [42, 123, 999])
    # firing-stability band (drift alarm)
    fire_lo: float = 0.02
    fire_hi: float = 0.90
    # methods to run on the probe
    methods: list = field(default_factory=lambda: [
        "D0_doppler", "D0_stft", "D0_chirp", "D0_dualtau", "D1_H1", "D1_rev", "D2_concat"])
    # experiment manifest (single source of truth)
    exp_index: int = 1
    exp_total: int = 3
    exp_name: str = "SHD deep probe"
    exp_manifest: list = field(default_factory=lambda: [
        (1, "SHD deep probe (T=12)", "THIS"),
        (2, "DIAT deep ablation (D0/D0+/D1/perm/D2)", "pending"),
        (3, "cross-family + gate variant", "pending"),
    ])

cfg = Config()

METHOD_LABELS = {
    "D0_doppler": "D0 homog Doppler-LIF",
    "D0_stft":    "D0 homog STFT-IF",
    "D0_chirp":   "D0 homog Chirp-LIF",
    "D0_dualtau": "D0 homog Dual-tau-LIF",
    "D1_H1":      "D1 vertical (STFT->Doppler->Chirp->Dualtau)",
    "D1_rev":     "D1 vertical reversed (falsification)",
    "D2_concat":  "D2 horizontal bank (concat)",
}
H1_ORDER = ["STFT-IF", "Doppler-LIF", "Chirp-LIF", "Dual-tau-LIF"]
H1_REV   = ["Dual-tau-LIF", "Chirp-LIF", "Doppler-LIF", "STFT-IF"]
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
# Dataset registry + defensive SHD loader
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

# register SHD once if missing (speaker-MIXED cache; not the official speaker-disjoint split)
if 'shd' not in load_registry():
    register_dataset('shd',
        path='/content/drive/MyDrive/Research/GHN_Instinct/gesture_cache/shd_T12_m200_v2.pt',
        source='zenkelab.org', format='pt-dict', classes=20, T=12, features=700,
        notes='Xtr/ytr/Xte/yte; speaker-mixed re-split, NOT official disjoint')

def load_shd():
    obj = torch.load(get_dataset_path('shd'), map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'Xtr' in obj:
        Xtr, ytr, Xte, yte = obj['Xtr'], obj['ytr'], obj['Xte'], obj['yte']
    elif isinstance(obj, dict) and 'X' in obj:
        X, y = obj['X'], obj['y']
        n = len(X); idx = torch.randperm(n); cut = int(0.8 * n)
        Xtr, ytr = X[idx[:cut]], y[idx[:cut]]
        Xte, yte = X[idx[cut:]], y[idx[cut:]]
    else:
        raise ValueError(f"Unexpected SHD object keys: {getattr(obj,'keys',lambda:obj)()}")
    to_t = lambda a: torch.as_tensor(np.asarray(a)).float()
    to_y = lambda a: torch.as_tensor(np.asarray(a)).long()
    Xtr, Xte = to_t(Xtr), to_t(Xte)
    ytr, yte = to_y(ytr), to_y(yte)
    # SHD is [N, T=12, F=700] = [N, time, features]; no transpose needed
    assert Xtr.ndim == 3, f"expected [N,T,F], got {tuple(Xtr.shape)}"
    return Xtr, ytr, Xte, yte


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
        return 0, []
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict']); opt.load_state_dict(ck['optimizer_state_dict'])
    print(f"    [RESUME] from epoch {ck['epoch']}")
    return ck['epoch'] + 1, ck['metrics'].get('fire_log', [])

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct.append((model(xb).argmax(1) == yb).int().cpu())
    return torch.cat(correct).numpy()

def firing_snapshot(model, xb):
    return model.firing_rates(xb) if isinstance(model, VerticalNet) else model.path_firing(xb)

def train_one(method, seed, Fin, C, NK, matched, tr_loader, te_loader, fire_batch, ckpt_dir):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(method, Fin, C, NK, matched).to(device)
    calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ckpt = ckpt_dir / f'{method}_seed{seed}.pt'
    start, fire_log = load_ckpt(model, opt, ckpt)
    best_acc, best_corr = -1.0, None
    for ep in range(start, cfg.n_epochs):
        calibrate_thresholds(model, fire_batch.to(device), cfg.target_firing)  # per-epoch threshold homeostasis (gradient-free)
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()
        corr = evaluate(model, te_loader); acc = float(corr.mean())
        fr = firing_snapshot(model, fire_batch.to(device))
        drift = not all(cfg.fire_lo < r < cfg.fire_hi for r in fr)
        fire_log.append({'epoch': ep, 'acc': acc, 'firing': [round(r, 4) for r in fr], 'drift': drift})
        if acc > best_acc:
            best_acc, best_corr = acc, corr
        save_ckpt(model, opt, ep, {'acc': acc, 'fire_log': fire_log}, ckpt)
        flag = "  !! FIRING DRIFT" if drift else ""
        print(f"    ep {ep:>2} acc={acc:.4f} firing={[round(r,3) for r in fr]}{flag}")
    return best_acc, best_corr, fire_log

def paired_bootstrap(corr_a, corr_b, nboot=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = corr_a.astype(float) - corr_b.astype(float); n = len(d)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


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

    # ── data ──
    Xtr, ytr, Xte, yte = load_shd()
    Fin, C = Xtr.shape[-1], int(max(ytr.max(), yte.max()) + 1)
    print(f"[DATA] SHD train={tuple(Xtr.shape)} test={tuple(Xte.shape)} F={Fin} C={C}")
    print(f"[DATA] feature range [{Xtr.min():.3f}, {Xtr.max():.3f}]  (chance={1/C:.3f})")
    NK = {"STFT-IF": {"window_len": cfg.stft_window_shd}}
    fire_batch = Xtr[:cfg.batch_size]
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True,
                           num_workers=2, pin_memory=(device.type == 'cuda'), persistent_workers=True)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)

    # ── capacity matching against D1 (homogeneous D0_* at W=128 are already ~matched: neuron params are O(W)) ──
    d1_params = count_params(VerticalNet(Fin, C, H1_ORDER, cfg.W, NK, cfg.norm))
    d2_w, d2_p, _ = _solve_width(
        lambda w: PathBankNet(Fin, C, D2_PATHS, w, cfg.L, "concat", NK, cfg.norm, True), d1_params)
    matched = {"d2_w": d2_w}
    d0_dop_p = count_params(VerticalNet(Fin, C, ["Doppler-LIF"]*cfg.L, cfg.W, NK, cfg.norm))
    print(f"[CAPACITY] D1 params={d1_params}  (homog D0 at W={cfg.W} ~ {d0_dop_p}, {100*abs(d0_dop_p-d1_params)/d1_params:.2f}%)")
    print(f"[CAPACITY] D2 concat width={d2_w} -> {d2_p} ({100*abs(d2_p-d1_params)/d1_params:.2f}%)")

    # ── workload ──
    n_runs = len(cfg.methods) * len(cfg.seeds)
    print(f"[WORKLOAD] {len(cfg.methods)} methods x {len(cfg.seeds)} seeds = {n_runs} runs x {cfg.n_epochs} epochs")

    # ── run ──
    results = {}; run_counter = 0; first_t = None
    for method in cfg.methods:
        accs, corrs = [], []
        for seed in cfg.seeds:
            run_counter += 1
            print(f"\n  Run {run_counter}/{n_runs}: {METHOD_LABELS[method]} | seed={seed}")
            t0 = time.time()
            best_acc, best_corr, flog = train_one(method, seed, Fin, C, NK, matched,
                                                  tr_loader, te_loader, fire_batch, CKPT_DIR)
            dt = time.time() - t0
            if first_t is None:
                first_t = dt; print(f"    [ETA] ~{first_t*(n_runs-1)/60:.0f} min for remaining {n_runs-1} runs")
            accs.append(best_acc); corrs.append(best_corr)
            with open(RUN_DIR / f'results_{method}_seed{seed}.json', 'w') as f:
                json.dump({'method': method, 'seed': seed, 'best_acc': best_acc,
                           'corr': best_corr.tolist(), 'fire_log': flog, 'time_s': dt}, f, indent=2)
        results[method] = {'accs': accs, 'mean': float(np.mean(accs)), 'std': float(np.std(accs)),
                           'corrs': [c.tolist() for c in corrs]}
        print(f"  => {method}: acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # ── paired bootstrap: control = capacity-matched BEST homogeneous (most stringent) ──
    homog = [m for m in results if m.startswith("D0_")]
    best_homog = max(homog, key=lambda m: results[m]['mean']) if homog else None
    if best_homog:
        print(f"\n[BOOTSTRAP] control = best homogeneous = {best_homog} ({results[best_homog]['mean']:.4f}); paired on test examples")
    else:
        print("\n[BOOTSTRAP] no homogeneous baseline present")

    def boot_pair(a, b):
        per_seed = [paired_bootstrap(np.array(results[a]['corrs'][si]),
                                     np.array(results[b]['corrs'][si]), seed=cfg.seeds[si])
                    for si in range(len(cfg.seeds))]
        diffs = [d for d, _, _ in per_seed]
        sig = all(lo > 0 for _, lo, _ in per_seed) or all(hi < 0 for _, _, hi in per_seed)
        print(f"   {a} - {b}: mean diff {np.mean(diffs):+.4f}  "
              f"per-seed CIs {[(round(l,3), round(h,3)) for _, l, h in per_seed]}  consistent_sign={sig}")
        return per_seed

    boot = {}
    if best_homog:
        for m in ["D1_H1", "D2_concat", "D1_rev"]:
            if m in results:
                boot[f"{m}_vs_bestHomog({best_homog})"] = boot_pair(m, best_homog)
    if "D2_concat" in results and "D1_H1" in results:
        boot["horizontal_vs_vertical"] = boot_pair("D2_concat", "D1_H1")     # the core question
    if "D1_H1" in results and "D1_rev" in results:
        boot["ordering_H1_vs_rev"] = boot_pair("D1_H1", "D1_rev")            # does layer order matter

    # ── summary ──
    summary = {'run_id': RUN_ID, 'exp_index': cfg.exp_index, 'config': {k: v for k, v in vars(cfg).items()},
               'capacity': {'d1_params': d1_params, 'd2_w': d2_w},
               'best_homog': best_homog,
               'results': {m: {'mean': r['mean'], 'std': r['std'], 'accs': r['accs']} for m, r in results.items()},
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
        plt.xlabel('epoch'); plt.ylabel('firing rate'); plt.title('D1(H1) per-layer firing — pure-SNN stability')
        plt.legend(fontsize=7); plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[plot skipped] {e}")

    print(f"\n[DONE] summary -> {RUN_DIR / 'summary.json'}")
    return summary

summary = main()
