# ============================================================================
# A cross-domain double dissociation of neuron-task matching in pure
# spiking networks  --  Dikmen & Karadag
#
# Verified, locked experiment script. The body below is the exact code that
# produced the paper numbers; it is kept self-contained for Colab paste-and-run.
# Requires the dikmen-spiking-neurons library (NeuronRegistry). See README.
#
# Cross-domain dissociation — SHD audio per-neuron homogeneous bake-off
# Paper: Section 4.3 (mixed-split half of Figure 4). Tests whether the neuron ranking
# changes across domains. Pre-registered prediction: Doppler-LIF is NOT top on audio.
# Source cell: "Shd deep probe" notebook, cell 4.
# ============================================================================

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-DOMAIN DISSOCIATION  ·  SHD (audio) per-neuron homogeneous bake-off
# Counterpart to the DIAT-µSAT ablation. Tests whether the neuron RANKING changes
# across domains: on DIAT (radar, frequency-structured) Doppler-LIF won; the
# pre-registered prediction is that on SHD (audio, temporal) Doppler-LIF is NOT
# the top neuron and a temporally-matched neuron (primary: Dual-tau-LIF) wins.
# Double dissociation holds iff  SHD-winner ≠ Doppler  while  DIAT-winner = Doppler.
#
# Controlled comparison (NOT HPO): identical config across neurons, matched to the
# DIAT ablation (L=4, W=128, target_firing=0.15, val-smoothed macro-F1, 3 seeds).
# Selection machinery identical to DIAT (init calibration, smoothed-val, test once).
#
# NOTE: this SHD cache is speaker-MIXED (not the official speaker-disjoint split),
# so absolute accuracy is NOT comparable to SHD literature. The dissociation only
# uses within-split neuron RANKING, which the split choice does not affect.
#
# Single paste-and-run Colab cell. SQLite-free; checkpoint + results.json resume.
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
    # candidates: full registry + vanilla baseline. DIAT-overlap set (direct crossover)
    # = {Doppler-LIF, Chirp-LIF, STFT-IF, Dual-tau-LIF}.
    NEURONS = ["Doppler-LIF", "Dual-tau-LIF", "Phase-LIF", "CrossInhib-LIF",
               "Chirp-LIF", "Beam-IF", "STFT-IF", "Vanilla-LIF"]
    L            = 4
    W            = 128
    TARGET_FIRING = 0.15
    NORM         = "none"          # pure SNN
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
    if IN_COLAB:
        PROJECT = "/content/drive/MyDrive/Research/NISAC_DeepHetero"
    else:
        PROJECT = "/home/claude/shd_run"

cfg = CFG()

if os.environ.get("REDUCED") == "1":
    cfg.NEURONS = ["Doppler-LIF", "Dual-tau-LIF", "Vanilla-LIF"]
    cfg.L = 2; cfg.W = 24; cfg.MAX_EPOCHS = 6; cfg.PATIENCE = 4; cfg.VAL_SMOOTH = 2
    cfg.SEEDS = [42]

# DIAT homogeneous ranking (locked ablation, macro-F1) for the side-by-side crossover
DIAT_HOMOG = {"Doppler-LIF": 0.8719, "Dual-tau-LIF": 0.8598,
              "Chirp-LIF": 0.6971, "STFT-IF": 0.1850}

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
RUN_ID  = os.environ.get("RUN_ID", "shd_dissoc")
RUN_DIR = Path(cfg.PROJECT) / "dissoc" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = RUN_DIR / "ckpt"; CKPT_DIR.mkdir(exist_ok=True)
print(f"[HW] device={device}" + (f"  gpu={torch.cuda.get_device_name(0)}" if device=='cuda' else ""), flush=True)
print(f"[RUN] {RUN_DIR}", flush=True)

# ── matched Vanilla-LIF baseline (same surrogate/interface, plain dynamics) ─────
class VanillaLIF(BaseNeuron):
    _family = "lif"
    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        return spk, {"mem": mem * (1.0 - spk)}
NeuronRegistry._all["Vanilla-LIF"] = VanillaLIF

# ══════════════════════════════════════════════════════════════════════════════
# SHD loader  (verified; .pt dict [N,T=12,F=700], 20 classes, speaker-mixed cache)
# ══════════════════════════════════════════════════════════════════════════════
REGISTRY_PATH = Path(cfg.PROJECT) / "datasets.json"
SHD_PATH = "/content/drive/MyDrive/Research/GHN_Instinct/gesture_cache/shd_T12_m200_v2.pt"

def load_shd():
    if not IN_COLAB:                      # sandbox: synthetic SHD-shaped data
        g = torch.Generator().manual_seed(0)
        N, T, Fin, C = 240, 12, 700, 20
        X = torch.rand(N, T, Fin, generator=g)
        y = torch.randint(0, C, (N,), generator=g)
        return X[:190], y[:190], X[190:], y[190:]
    obj = torch.load(SHD_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "Xtr" in obj:
        Xtr, ytr, Xte, yte = obj["Xtr"], obj["ytr"], obj["Xte"], obj["yte"]
    elif isinstance(obj, dict) and "X" in obj:
        X, y = obj["X"], obj["y"]
        n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)); cut = int(0.8*n)
        Xtr, ytr, Xte, yte = X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]
    else:
        raise ValueError(f"Unexpected SHD keys: {list(obj.keys())}")
    to_t = lambda a: torch.as_tensor(np.asarray(a)).float()
    to_y = lambda a: torch.as_tensor(np.asarray(a)).long()
    Xtr, Xte, ytr, yte = to_t(Xtr), to_t(Xte), to_y(ytr), to_y(yte)
    assert Xtr.ndim == 3, f"expected [N,T,F], got {tuple(Xtr.shape)}"
    return Xtr, ytr, Xte, yte

def _strat_val(Xa, ya, frac, seed):
    try:
        from sklearn.model_selection import train_test_split
        A, B, ya2, yb2 = train_test_split(Xa.numpy(), ya.numpy(), test_size=frac,
                                           stratify=ya.numpy(), random_state=seed)
        return (torch.as_tensor(A).float(), torch.as_tensor(ya2).long(),
                torch.as_tensor(B).float(), torch.as_tensor(yb2).long())
    except Exception:
        rng = np.random.default_rng(seed); ia, ib = [], []
        for c in torch.unique(ya).tolist():
            ci = np.where(ya.numpy() == c)[0]; rng.shuffle(ci); k = int((1-frac)*len(ci))
            ia += list(ci[:k]); ib += list(ci[k:])
        return Xa[ia], ya[ia], Xa[ib], ya[ib]

Xtrall, ytrall, Xte, yte = load_shd()
# standardize with train stats (global), same as DIAT pipeline
mu, sd = float(Xtrall.mean()), float(Xtrall.std() + 1e-6)
Xtrall = (Xtrall - mu)/sd; Xte = (Xte - mu)/sd
Xtr, ytr, Xva, yva = _strat_val(Xtrall, ytrall, cfg.VAL_FRAC, 0)
C = int(max(ytr.max(), yte.max()) + 1); Fin = Xtr.shape[-1]; Tlen = Xtr.shape[1]
print(f"[DATA] SHD train={tuple(Xtr.shape)} val={tuple(Xva.shape)} test={tuple(Xte.shape)} "
      f"T={Tlen} F={Fin} C={C} chance={1/C:.3f}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Harness (inline) — identical to the verified DIAT cells
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
# Run all neurons × seeds  (resume via results.json)
# ══════════════════════════════════════════════════════════════════════════════
RES_PATH = RUN_DIR / "results.json"
res = json.loads(RES_PATH.read_text()) if RES_PATH.exists() else {}
print("\n" + "═"*78 + "\n SHD per-neuron bake-off (homogeneous, matched config)\n" + "═"*78, flush=True)
for neuron in cfg.NEURONS:
    for seed in cfg.SEEDS:
        key = f"{neuron}:{seed}"
        if key in res: continue
        print(f"\n[{neuron} seed {seed}]", flush=True)
        r = train_one(neuron, seed)
        res[key] = r; RES_PATH.write_text(json.dumps(res))
        print(f"   -> {key}: F1={r['f1']:.4f} acc={r['acc']:.4f} fire={r['firing']:.3f} "
              f"@ep{r['best_epoch']} params={r['params']}", flush=True)

# ══════════════════════════════════════════════════════════════════════════════
# Aggregate ranking + dissociation crossover + bootstrap
# ══════════════════════════════════════════════════════════════════════════════
def agg(neuron):
    ks = [f"{neuron}:{s}" for s in cfg.SEEDS if f"{neuron}:{s}" in res]
    f1 = [res[k]["f1"] for k in ks]; ac = [res[k]["acc"] for k in ks]; fr = [res[k]["firing"] for k in ks]
    return ks, f1, ac, fr

print("\n" + "═"*78 + "\n SHD RANKING  (macro-F1, mean±std over seeds)\n" + "═"*78, flush=True)
ranking = []
for neuron in cfg.NEURONS:
    ks, f1, ac, fr = agg(neuron)
    if not ks: continue
    ranking.append((neuron, float(np.mean(f1)), float(np.std(f1)), float(np.mean(ac)), float(np.mean(fr))))
ranking.sort(key=lambda x: x[1], reverse=True)
for i,(n,f,s,a,fr) in enumerate(ranking):
    tag = "  <-- DIAT winner" if n == "Doppler-LIF" else ""
    print(f"   {i+1}. {n:16s} F1 {f:.4f}±{s:.4f}  acc {a:.4f}  fire {fr:.3f}{tag}", flush=True)

shd_winner = ranking[0][0] if ranking else None
print("\n" + "═"*78 + "\n DOUBLE DISSOCIATION  (DIAT-overlap neurons)\n" + "═"*78, flush=True)
print(f"   {'neuron':16s}   DIAT-F1    SHD-F1", flush=True)
for n in ["Doppler-LIF", "Dual-tau-LIF", "Chirp-LIF", "STFT-IF"]:
    shd = next((r[1] for r in ranking if r[0]==n), None)
    d = DIAT_HOMOG.get(n)
    print(f"   {n:16s}   {d:.4f}    {shd:.4f}" if shd is not None else f"   {n:16s}   {d:.4f}    (n/a)", flush=True)
print(f"\n   DIAT winner = Doppler-LIF   |   SHD winner = {shd_winner}", flush=True)
dissociation = (shd_winner is not None and shd_winner != "Doppler-LIF")
print(f"   DISSOCIATION {'CONFIRMED' if dissociation else 'NOT confirmed'} "
      f"(SHD winner {'≠' if dissociation else '=='} Doppler)", flush=True)

print("\n[BOOTSTRAP] paired on SHD test, per seed (consistent = same sign all seeds)", flush=True)
boot = {}
pairs = []
if shd_winner and shd_winner != "Doppler-LIF": pairs.append((shd_winner, "Doppler-LIF"))
pairs += [("Doppler-LIF", "Vanilla-LIF"), ("Dual-tau-LIF", "Doppler-LIF")]
seen = set()
for a, b in pairs:
    if (a,b) in seen or a == b: continue
    seen.add((a,b))
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

summary = {"run_dir": str(RUN_DIR), "config": {"L": cfg.L, "W": cfg.W, "target_firing": cfg.TARGET_FIRING,
           "seeds": cfg.SEEDS, "max_epochs": cfg.MAX_EPOCHS}, "ranking": ranking,
           "shd_winner": shd_winner, "dissociation": dissociation, "diat_homog": DIAT_HOMOG,
           "bootstrap": boot, "timestamp": datetime.now().isoformat()}
(RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\n[DONE] summary -> {RUN_DIR / 'summary.json'}", flush=True)
