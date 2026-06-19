# ============================================================================
# A cross-domain double dissociation of neuron-task matching in pure
# spiking networks  --  Dikmen & Karadag
#
# Verified, locked experiment script. The body below is the exact code that
# produced the paper numbers; it is kept self-contained for Colab paste-and-run.
# Requires the dikmen-spiking-neurons library (NeuronRegistry). See README.
#
# Close the 2x2 — DIAT-uSAT (radar) bake-off for the cross neurons
# Paper: Section 4.3 / Figure 5. Runs {Doppler, CrossInhib, Phase} homogeneously on
# DIAT under the exact ablation config to complete the double-dissociation square.
# Source cell: "Shd deep probe" notebook, cell 5.
# ============================================================================

# ══════════════════════════════════════════════════════════════════════════════
# CLOSE THE 2×2  ·  DIAT-µSAT (radar) homogeneous bake-off for the cross neurons
# Completes the double-dissociation square. SHD (audio) winner was CrossInhib-LIF
# (Doppler 6th of 8). This runs {Doppler, CrossInhib, Phase} homogeneously on DIAT
# under the EXACT ablation config to show the mirror image: Doppler wins DIAT and
# CrossInhib falls below it. Doppler is re-run both for the paired bootstrap (same
# test split) and as a harness-consistency check (must reproduce ~0.872).
#
# 2×2 double dissociation CONFIRMED iff:
#     Doppler_DIAT > CrossInhib_DIAT   AND   CrossInhib_SHD > Doppler_SHD
#
# Config identical to the DIAT ablation and the SHD cell (L=4, W=128, fire=0.15,
# val-smoothed macro-F1, 3 seeds). Single paste-and-run Colab cell, resumable.
# ══════════════════════════════════════════════════════════════════════════════
import os, json, math, copy, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False

# ────────────────────────────── CONFIG ────────────────────────────────────────
class CFG:
    NEURONS = ["Doppler-LIF", "CrossInhib-LIF", "Phase-LIF"]   # Doppler re-run for paired bootstrap
    L            = 4
    W            = 128
    TARGET_FIRING = 0.15
    NORM         = "none"
    MAX_EPOCHS   = 200
    PATIENCE     = 20
    VAL_SMOOTH   = 5
    VAL_FRAC     = 0.15
    LR           = 1e-3
    BATCH        = 128
    SELECT_METRIC = "macro_f1"
    SEEDS        = [42, 123, 999]
    USE_TQDM     = True
    HEARTBEAT_EVERY = 5
    PROJECT = "/content/drive/MyDrive/Research/NISAC_DeepHetero" if IN_COLAB else "/home/claude/diat_run"

cfg = CFG()
if os.environ.get("REDUCED") == "1":
    cfg.NEURONS = ["Doppler-LIF", "CrossInhib-LIF", "Phase-LIF"]
    cfg.L = 2; cfg.W = 24; cfg.MAX_EPOCHS = 6; cfg.PATIENCE = 4; cfg.VAL_SMOOTH = 2
    cfg.SEEDS = [42]

# locked DIAT ablation homogeneous (macro-F1) — Doppler value is the reproduction target
DIAT_HOMOG = {"Doppler-LIF": 0.8719, "Dual-tau-LIF": 0.8598, "Chirp-LIF": 0.6971, "STFT-IF": 0.1850}
# measured SHD homogeneous ranking (this study) — the cross-domain reference
SHD_HOMOG = {"CrossInhib-LIF": 0.7405, "Vanilla-LIF": 0.7030, "Dual-tau-LIF": 0.6943,
             "Chirp-LIF": 0.6624, "Beam-IF": 0.6385, "Doppler-LIF": 0.6005,
             "STFT-IF": 0.2439, "Phase-LIF": 0.1430}

if IN_COLAB:
    os.system("pip install -q git+https://github.com/DrCanD/dikmen-spiking-neurons.git tqdm")
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import dikmen_neurons as D
from dikmen_neurons import BaseNeuron, spike_hard, NeuronRegistry
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

device = "cuda" if torch.cuda.is_available() else "cpu"
RUN_ID  = os.environ.get("RUN_ID", "diat_2x2")
RUN_DIR = Path(cfg.PROJECT) / "dissoc" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "ckpt"; CKPT_DIR.mkdir(exist_ok=True)
print(f"[HW] device={device}" + (f"  gpu={torch.cuda.get_device_name(0)}" if device=='cuda' else ""), flush=True)
print(f"[RUN] {RUN_DIR}", flush=True)

# Vanilla baseline registered for parity (not in NEURONS by default, but harmless)
class VanillaLIF(BaseNeuron):
    _family = "lif"
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        return spk, {"mem": mem * (1.0 - spk)}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF

# ══════════════════════════════════════════════════════════════════════════════
# DIAT loader (verbatim from the verified ablation: [N,64,64]=[N,time,Doppler],
# 80/20 stratified seed=0, val carved from trainval, train-only standardization)
# ══════════════════════════════════════════════════════════════════════════════
DIAT_PATH = "/content/drive/MyDrive/NISAC/data/DIAT_uSAT/processed"

def _diat_xy():
    if not IN_COLAB:                       # sandbox: synthetic DIAT-shaped data
        g = np.random.default_rng(0)
        return g.standard_normal((240, 64, 64)).astype(np.float32), g.integers(0, 6, 240).astype(np.int64)
    X = np.load(Path(DIAT_PATH) / "X.npy").astype(np.float32)   # [N,64,64]
    y = np.load(Path(DIAT_PATH) / "y.npy").astype(np.int64)
    return X, y

def load_diat():
    X, y = _diat_xy()
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
    Xtv, ytv, Xte, yte = strat_split(X, y, 0.20, 0)
    Xtr, ytr, Xva, yva = strat_split(Xtv, ytv, cfg.VAL_FRAC / 0.80, 0)
    mu, sd = float(Xtr.mean()), float(Xtr.std() + 1e-6)
    norm = lambda a: torch.as_tensor((a - mu) / sd).float()
    yy   = lambda a: torch.as_tensor(a).long()
    return norm(Xtr), yy(ytr), norm(Xva), yy(yva), norm(Xte), yy(yte)

Xtr, ytr, Xva, yva, Xte, yte = load_diat()
C = int(max(ytr.max(), yte.max()) + 1); Fin = Xtr.shape[-1]; Tlen = Xtr.shape[1]
print(f"[DATA] DIAT train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)} "
      f"T={Tlen} F={Fin} C={C} chance={1/C:.3f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Harness (inline) — identical to the verified DIAT/SHD cells
# ══════════════════════════════════════════════════════════════════════════════
def _make_norm(kind, width):
    if kind == "layer": return nn.LayerNorm(width)
    if kind == "batch": return nn.BatchNorm1d(width)
    return nn.Identity()

class FeatureStack(nn.Module):
    def __init__(self, in_features, layer_types, width, neuron_kwargs=None, norm="none"):
        super().__init__()
        neuron_kwargs = neuron_kwargs or {}
        dims = [in_features] + [width]*len(layer_types)
        self.projs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(layer_types))])
        self.norms = nn.ModuleList([_make_norm(norm, width) for _ in layer_types])
        self.neurons = nn.ModuleList(
            [NeuronRegistry.create(t, size=width, **neuron_kwargs.get(t, {})) for t in layer_types])
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
    def forward(self, x):
        feat, _ = self.stack(x); return self.readout(feat)
    def firing_rates(self, x): return self.stack.layer_firing(x)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

@torch.no_grad()
def calibrate_thresholds(model, x, target, iters=30):
    h = x
    for proj, norm, neuron in zip(model.stack.projs, model.stack.norms, model.stack.neurons):
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
    return model

def accuracy(p, t): return float((p == t).mean())
def macro_f1(p, t, C):
    f1s = []
    for c in range(C):
        tp = int(np.sum((p==c)&(t==c))); fp = int(np.sum((p==c)&(t!=c))); fn = int(np.sum((p!=c)&(t==c)))
        pr = tp/(tp+fp) if (tp+fp) else 0.0; rc = tp/(tp+fn) if (tp+fn) else 0.0
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) else 0.0)
    return float(np.mean(f1s))
METRICS = {"macro_f1": lambda p,t,C: macro_f1(p,t,C), "accuracy": lambda p,t,C: accuracy(p,t)}

@torch.no_grad()
def evaluate(model, X, Y, batch):
    model.eval(); preds = []
    for i in range(0, len(X), batch):
        preds.append(model(X[i:i+batch].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy(), Y.numpy()

def firing_mean(model, xb): return float(np.mean(model.firing_rates(xb)))

def paired_bootstrap(pa, pb, trues, mfn, C, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(trues); boots = np.empty(nboot)
    base = mfn(pa, trues, C) - mfn(pb, trues, C)
    for k in range(nboot):
        i = rng.integers(0, n, n)
        boots[k] = mfn(pa[i], trues[i], C) - mfn(pb[i], trues[i], C)
    return float(base), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

# ══════════════════════════════════════════════════════════════════════════════
# Train one homogeneous net (val-smoothed selection, test once at best epoch)
# ══════════════════════════════════════════════════════════════════════════════
def train_one(neuron, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    fire_batch = Xtr[:256].to(device)
    model = VerticalNet(Fin, C, [neuron]*cfg.L, cfg.W, {}, cfg.NORM).to(device)
    calibrate_thresholds(model, fire_batch, cfg.TARGET_FIRING)
    n_params = count_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    sel = METRICS[cfg.SELECT_METRIC]

    ckpt = CKPT_DIR / f"{neuron.replace('/','_')}_s{seed}.pt"
    start, m = 0, {}
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start, m = ck["epoch"]+1, ck["m"]
    val_hist = m.get("val_hist", []); best_val = m.get("best_val", -1.0)
    best_epoch = m.get("best_epoch", -1); pc = m.get("pc", 0); best = m.get("best", None)

    use_bar = cfg.USE_TQDM and _HAS_TQDM
    it = tqdm(range(start, cfg.MAX_EPOCHS), desc=f"{neuron} s{seed}", leave=False,
              initial=start, total=cfg.MAX_EPOCHS) if use_bar else range(start, cfg.MAX_EPOCHS)
    for ep in it:
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg.BATCH):
            b = perm[i:i+cfg.BATCH]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(Xtr[b].to(device)), ytr[b].to(device)).backward(); opt.step()
        vp, vt = evaluate(model, Xva, yva, cfg.BATCH); val_hist.append(sel(vp, vt, C))
        v_smooth = float(np.mean(val_hist[-cfg.VAL_SMOOTH:]))
        tp, tt = evaluate(model, Xte, yte, cfg.BATCH)
        t_f1 = macro_f1(tp, tt, C); t_acc = accuracy(tp, tt)
        improved = v_smooth > best_val + 1e-4
        if improved:
            best_val, best_epoch, pc = v_smooth, ep, 0
            best = {"f1": t_f1, "acc": t_acc, "firing": firing_mean(model, fire_batch),
                    "preds": tp.tolist(), "trues": tt.tolist()}
        else:
            pc += 1
        torch.save({"epoch": ep, "model": copy.deepcopy(model.state_dict()), "opt": opt.state_dict(),
                    "m": {"val_hist": val_hist, "best_val": best_val, "best_epoch": best_epoch,
                          "pc": pc, "best": best}}, ckpt)
        cur = best["firing"] if best else 0.0
        if use_bar:
            it.set_postfix(val=f"{v_smooth:.3f}", f1=f"{t_f1:.3f}", fire=f"{cur:.2f}", pat=pc)
        elif (ep % cfg.HEARTBEAT_EVERY == 0) or improved:
            print(f"      ep{ep:>3}/{cfg.MAX_EPOCHS} val={v_smooth:.4f} f1={t_f1:.4f} "
                  f"fire={cur:.3f} pat={pc}{'  *' if improved else ''}", flush=True)
        if pc >= cfg.PATIENCE: break
    if use_bar: it.close()
    best["params"] = n_params; best["best_epoch"] = best_epoch
    return best

# ══════════════════════════════════════════════════════════════════════════════
# Run + aggregate + 2×2 closure + bootstrap
# ══════════════════════════════════════════════════════════════════════════════
RES_PATH = RUN_DIR / "results.json"
res = json.loads(RES_PATH.read_text()) if RES_PATH.exists() else {}
print("\n" + "═"*78 + "\n DIAT per-neuron bake-off (homogeneous, matched to ablation)\n" + "═"*78, flush=True)
for neuron in cfg.NEURONS:
    for seed in cfg.SEEDS:
        key = f"{neuron}:{seed}"
        if key in res: continue
        print(f"\n[{neuron} seed {seed}]", flush=True)
        r = train_one(neuron, seed)
        res[key] = r; RES_PATH.write_text(json.dumps(res))
        print(f"   -> {key}: F1={r['f1']:.4f} acc={r['acc']:.4f} fire={r['firing']:.3f} "
              f"@ep{r['best_epoch']} params={r['params']}", flush=True)

def agg(neuron):
    ks = [f"{neuron}:{s}" for s in cfg.SEEDS if f"{neuron}:{s}" in res]
    return ks, [res[k]["f1"] for k in ks], [res[k]["acc"] for k in ks], [res[k]["firing"] for k in ks]

print("\n" + "═"*78 + "\n DIAT RANKING  (macro-F1, mean±std over seeds)\n" + "═"*78, flush=True)
diat_now = {}
for neuron in cfg.NEURONS:
    ks, f1, ac, fr = agg(neuron)
    if not ks: continue
    diat_now[neuron] = float(np.mean(f1))
    print(f"   {neuron:16s} F1 {np.mean(f1):.4f}±{np.std(f1):.4f}  acc {np.mean(ac):.4f}  fire {np.mean(fr):.3f}", flush=True)

# harness-consistency check on the Doppler reproduction
dop_repro = diat_now.get("Doppler-LIF")
if dop_repro is not None:
    print(f"\n[CHECK] Doppler-LIF DIAT reproduction: {dop_repro:.4f}  (locked ablation 0.8719, "
          f"Δ={dop_repro-0.8719:+.4f})", flush=True)

# ── the 2×2 ────────────────────────────────────────────────────────────────────
print("\n" + "═"*78 + "\n DOUBLE DISSOCIATION 2×2  (winner of each domain)\n" + "═"*78, flush=True)
print(f"   {'neuron':16s}   DIAT-F1    SHD-F1", flush=True)
for n in ["Doppler-LIF", "CrossInhib-LIF"]:
    d = diat_now.get(n, DIAT_HOMOG.get(n)); s = SHD_HOMOG.get(n)
    print(f"   {n:16s}   {d:.4f}    {s:.4f}", flush=True)
ciP = diat_now.get("Phase-LIF")
if ciP is not None:
    print(f"   {'Phase-LIF':16s}   {ciP:.4f}    {SHD_HOMOG['Phase-LIF']:.4f}", flush=True)

dopD, ciD = diat_now.get("Doppler-LIF"), diat_now.get("CrossInhib-LIF")
ciS, dopS = SHD_HOMOG["CrossInhib-LIF"], SHD_HOMOG["Doppler-LIF"]
square = (dopD is not None and ciD is not None and dopD > ciD and ciS > dopS)
print(f"\n   DIAT: Doppler {dopD:.4f} vs CrossInhib {ciD:.4f}  -> "
      f"{'Doppler wins' if dopD>ciD else 'CrossInhib wins'}", flush=True)
print(f"   SHD : CrossInhib {ciS:.4f} vs Doppler {dopS:.4f}  -> "
      f"{'CrossInhib wins' if ciS>dopS else 'Doppler wins'}", flush=True)
print(f"   2×2 DOUBLE DISSOCIATION {'CONFIRMED' if square else 'NOT confirmed'} "
      f"(each domain won by its physics-matched neuron)", flush=True)

print("\n[BOOTSTRAP] paired on DIAT test, per seed (consistent = same sign all seeds)", flush=True)
boot = {}
for a, b in [("Doppler-LIF", "CrossInhib-LIF"), ("Doppler-LIF", "Phase-LIF")]:
    if not (any(f"{a}:{s}" in res for s in cfg.SEEDS) and any(f"{b}:{s}" in res for s in cfg.SEEDS)): continue
    for label, mfn in [("F1", METRICS["macro_f1"]), ("acc", METRICS["accuracy"])]:
        per = []
        for s in cfg.SEEDS:
            ka, kb = f"{a}:{s}", f"{b}:{s}"
            if ka in res and kb in res:
                per.append(paired_bootstrap(np.array(res[ka]["preds"]), np.array(res[kb]["preds"]),
                                            np.array(res[ka]["trues"]), mfn, C))
        if not per: continue
        mean = float(np.mean([x[0] for x in per])); cis = [(round(l,3), round(h,3)) for _,l,h in per]
        consistent = all(l>0 for _,l,_ in per) or all(h<0 for _,_,h in per)
        boot[f"{a}-{b}[{label}]"] = {"mean": mean, "cis": cis, "consistent": consistent}
        print(f"   {a} - {b} [{label}]: mean {mean:+.4f}  CIs {cis}  consistent={consistent}", flush=True)

summary = {"run_dir": str(RUN_DIR), "diat_now": diat_now, "doppler_reproduction": dop_repro,
           "diat_homog_locked": DIAT_HOMOG, "shd_homog": SHD_HOMOG,
           "square_confirmed": bool(square), "bootstrap": boot, "timestamp": datetime.now().isoformat()}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[DONE] summary -> {RUN_DIR / 'summary.json'}", flush=True)
