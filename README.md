# A cross-domain double dissociation of neuron-task matching in pure spiking networks

[![DOI](https://zenodo.org/badge/1274450750.svg)](https://doi.org/10.5281/zenodo.20763581)

Experiment code for the paper of the same name (Dikmen & Karadağ, submitted to
*Neural Networks*, Elsevier, 2026). The central claim is that at matched capacity a
network whose single neuron type is matched to the task beats a heterogeneous mix of
neuron types, and that this advantage flips across domains: the neuron that wins on
micro-Doppler radar (DIAT-µSAT) is near the bottom on spoken-digit audio (Spiking
Heidelberg Digits), while another neuron shows the opposite pattern. That crossed
pattern is the double dissociation.

All networks here are pure spiking: the signal passed between layers is spikes, with
no batch or layer normalisation and no softmax gating inside the network. Per-type
firing thresholds are calibrated once at initialisation so no layer starts dead or
saturated, after which firing runs free and is logged as an efficiency metric.

## Layout

```
src/mstask/            installable package — the shared, verified-equivalent harness
  __init__.py            ensures the spiking-neuron library is importable
  core.py                models, threshold calibration, capacity helpers, metrics, registry
experiments/           one runnable script per experiment (imports from mstask.core)
  01_shd_probe.py              harness probe (SHD T=12), not a paper table
  02_diat_ablation.py          HEADLINE matched-capacity ablation — Section 4.1, Figure 3
  03_best_model_search.py      best-model search — Section 4.2, Table 2
  04_best_model_search_full.py search + compact-CNN reference + op-count proxy — Table 2
  05_shd_bakeoff.py            SHD audio bake-off, speaker-mixed split — Section 4.3, Figure 4
  06_close_2x2.py              DIAT bake-off that closes the dissociation — Section 4.3, Figure 5
  07_shd_disjoint.py           SHD official speaker-disjoint re-run — Section 3.3, Table 3
results/               locked run summaries that back the paper tables (full per-seed scores)
figures/               notes on regenerating the manuscript figures from results/
```

The experiment scripts were factored out of the original single-cell Colab notebook
that produced the paper. The parts that are identical across all experiments (the
spiking models, threshold calibration, capacity solver, metrics and the dataset
registry) were lifted verbatim into `mstask.core` after an abstract-syntax-tree
comparison confirmed they differ only in formatting, never in computation. The parts
that genuinely differ between experiments — the significance test, the training loop,
the model builder and the data loaders — were left inside each script so the numbers
each one produces stay exactly what they were.

## Install

```bash
pip install -e .
```

This installs the `mstask` package and its dependencies, including the
physics-motivated spiking-neuron library
([dikmen-spiking-neurons](https://github.com/DrCanD/dikmen-spiking-neurons),
Zenodo [10.5281/zenodo.20110833](https://doi.org/10.5281/zenodo.20110833)), which
provides the Doppler, Chirp, Dual-tau, STFT, Phase, Beam and CrossInhib units through
`NeuronRegistry`. Importing `mstask` also installs that library from GitHub on first
use if it is missing, so the scripts also run as-is when pasted into a fresh Colab cell.

## Data

The datasets are not redistributed here.

- **DIAT-µSAT micro-Doppler** (radar): a folder with `X.npy` `[N, 64, 64]` and
  `y.npy`, six classes, spectrogram-as-sequence with `T = 64`.
- **Spiking Heidelberg Digits** (audio): twenty spoken digits from
  [zenkelab.org](https://zenkelab.org). Experiment 07 downloads the official
  speaker-disjoint split and bins it to `T = 12`; the mixed-split cache used by 01 and
  05 is a re-split `.pt` dictionary.

Point the scripts at your copies with environment variables (the author's Colab paths
are kept as defaults for provenance):

```bash
export RESEARCH_ROOT=/path/to/research        # run/output root
export DIAT_DATA=/path/to/DIAT_uSAT/processed # DIAT folder with X.npy + y.npy
export SHD_CACHE=/path/to/shd_T12_m200_v2.pt  # mixed-split SHD cache
```

## Run

```bash
python experiments/02_diat_ablation.py   # headline ablation
python experiments/07_shd_disjoint.py    # official speaker-disjoint audio ranking
```

Each script trains three seeds (42, 123, 999), selects on validation-smoothed
macro-F1, tests once at the best epoch, and writes a `summary.json` into its run
folder. The headline and disjoint scripts resume from a checkpoint if interrupted.
Set `REDUCED=1` for a fast reduced configuration where supported.

A note on the environment: the scripts were checked here with `py_compile` and a
static undefined-name pass, but a full end-to-end training run needs a GPU and the
datasets, so run `experiments/01_shd_probe.py` once to confirm the harness on your
machine before trusting a long run.

## Results

`results/` holds the verified run summaries with per-seed scores, so the means,
standard deviations and paired-bootstrap statements can be re-derived without
re-training. `shd_disjoint_summary.json` backs Table 3 (Dual-tau-LIF 0.671 top,
CrossInhib-LIF 0.635, Doppler-LIF sixth at 0.451; Spearman between the mixed and
disjoint rankings 0.76). `cnn_frontier_summary.json` backs Table 2 (Doppler-LIF
0.924 / 475k, Vanilla 0.857, Parametric 0.852, compact-CNN reference 0.932 / ~24k);
its energy fields are an operation-count proxy, not an energy claim.

## Citation

If you use this code, please cite the paper, this repository
(DOI [10.5281/zenodo.20763581](https://doi.org/10.5281/zenodo.20763581)) and the
neuron library; see `CITATION.cff`. Licensed under MIT.
