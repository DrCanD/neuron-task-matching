"""
mstask.core — shared, verified-equivalent harness for the neuron-task-matching study.

Every symbol here was extracted verbatim from the original experiment cells after an
AST comparison confirmed the seven cells differ only in formatting for these pieces
(line wrapping, variable names, ternary vs if/else), never in computation. The parts
that genuinely diverge between experiments (the significance test, the training loop,
the model builder and the data loaders) stay inside each experiment script.
"""
import os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from dikmen_neurons import NeuronRegistry
try:
    from sklearn.model_selection import train_test_split
except Exception:
    train_test_split = None

RESEARCH = Path(os.environ.get("RESEARCH_ROOT", "./data"))
REGISTRY_PATH = RESEARCH / "datasets.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- _make_norm  (canonical from cell2_best_search_A) ----
def _make_norm(kind, width):
    if kind == "layer": return nn.LayerNorm(width)
    if kind == "batch": return nn.BatchNorm1d(width)
    return nn.Identity()

# ---- FeatureStack  (canonical from cell1_diat_ablation) ----
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

# ---- VerticalNet  (canonical from cell6_shd_disjoint) ----
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

# ---- PathBankNet  (canonical from cell0_smoke) ----
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

# ---- count_params  (canonical from cell2_best_search_A) ----
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---- _solve_width  (canonical from cell1_diat_ablation) ----
def _solve_width(builder, target, wmax=400):
    best = None
    for w in range(2, wmax):
        p = count_params(builder(w)); d = abs(p - target)
        if best is None or d < best[2]:
            best = (w, p, d)
    return best

# ---- _calibrate_stack  (canonical from cell1_diat_ablation) ----
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

# ---- calibrate_thresholds  (canonical from cell1_diat_ablation) ----
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

# ---- accuracy  (canonical from cell1_diat_ablation) ----
def accuracy(preds, trues, C=None):
    return float((preds == trues).mean())

# ---- macro_f1  (canonical from cell1_diat_ablation) ----
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

# ---- firing_snapshot  (canonical from cell1_diat_ablation) ----
def firing_snapshot(model, xb):
    return model.firing_rates(xb) if isinstance(model, VerticalNet) else model.path_firing(xb)

# ---- spearman  (canonical from cell6_shd_disjoint) ----
def spearman(names, score_x, score_y):
    xs = np.array([score_x[n] for n in names], dtype=float)
    ys = np.array([score_y[n] for n in names], dtype=float)
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])

# ---- _strat_split  (canonical from cell6_shd_disjoint) ----
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

# ---- load_registry  (canonical from cell1_diat_ablation) ----
def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

# ---- register_dataset  (canonical from cell1_diat_ablation) ----
def register_dataset(name, path, **meta):
    reg = load_registry()
    reg[name] = {'path': str(path), 'registered': datetime.now().isoformat(), **meta}
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(reg, f, indent=2)

# ---- get_dataset_path  (canonical from cell1_diat_ablation) ----
def get_dataset_path(name):
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"Dataset '{name}' not in registry {list(reg.keys())}. Register it.")
    p = Path(reg[name]['path'])
    if not p.exists():
        raise FileNotFoundError(f"'{name}' registered at {p} but not found.")
    return p

