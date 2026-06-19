# A cross-domain double dissociation of neuron-task matching in pure spiking networks
[![DOI](https://zenodo.org/badge/1274450750.svg)](https://doi.org/10.5281/zenodo.20763581)

Experiment code for the paper *A cross-domain double dissociation of neuron-task
matching in pure spiking networks* by İsmail Can Dikmen and Teoman Karadağ
(submitted to *Neural Networks*, Elsevier).

The paper asks whether a spiking network should combine many neuron types or match
one type to the task. At matched capacity and under a held-out protocol, heterogeneity
does not beat the best task-matched homogeneous neuron on radar micro-Doppler, and a
double dissociation between radar (DIAT-µSAT) and audio (Spiking Heidelberg Digits)
identifies neuron-task matching, rather than heterogeneity, as the driver.

This repository holds the experiment harness: the pure-spiking architectures, the
threshold-calibration and training loop, the matched-capacity logic, the
multi-objective search, and the cross-domain bake-offs. The spiking neurons
themselves come from a separate library (see below).

## Spiking neuron library (dependency)

The seven physics-motivated neurons (Doppler, Chirp, Dual-tau, STFT, Phase, Beam,
CrossInhib) and the two LIF controls are the MS-IF family of the
`dikmen-spiking-neurons` library, exposed through `NeuronRegistry`. Each experiment
installs it on first run:

```
pip install git+https://github.com/DrCanD/dikmen-spiking-neurons.git
```

Library archive: Zenodo, doi:10.5281/zenodo.20110833 (v1.2.0).

## Repository layout

```
experiments/   seven runnable scripts, one per stage of the study
notebooks/      the original Colab notebook (outputs stripped)
results/        locked summary JSONs that back the paper tables
figures/        notes on figure reproduction
```

Every script under `experiments/` is the exact, self-contained code that produced the
reported numbers. The harness is intentionally repeated inside each script rather than
imported from a shared module, so that any single script can be pasted into one Colab
cell and run end-to-end, which is how the runs were executed. Each script is
checkpoint-resumable and writes a `summary.json` to its run folder.

## Experiments and where they appear in the paper

| Script | Paper | What it runs |
|---|---|---|
| `01_shd_deep_probe.py` | harness probe, not a table | Smoke, speed and firing-stability check on SHD (T=12) before the headline run |
| `02_diat_ablation.py` | Section 4.1, Figure 3 | Matched-capacity ablation on DIAT-µSAT: homogeneous (D0) vs vertical (D1) vs horizontal bank (D2) |
| `03_best_model_search.py` | Section 4.2, Table 2 | Multi-objective Optuna search over Doppler, the D2 bank and Vanilla-LIF |
| `04_best_model_search_full.py` | Section 4.2, Table 2 | Full search with Parametric-LIF, the compact-CNN reference and the operation-count proxy |
| `05_shd_bakeoff.py` | Section 4.3, Figure 4 (mixed split) | SHD audio per-neuron homogeneous bake-off on the speaker-mixed re-split |
| `06_close_2x2.py` | Section 4.3, Figure 5 | DIAT-µSAT bake-off for {Doppler, CrossInhib, Phase} to close the dissociation square |
| `07_shd_disjoint.py` | Section 3.3 and Table 3, Figure 4 | SHD official speaker-disjoint re-run; self-downloads SHD and bins to 12 frames |

## Architecture and protocol

The pure-spiking stack is depth `L = 4`, width `W = 128`, with no normalization
anywhere; the read-out is the temporal mean of the last-layer spikes followed by a
linear classifier, trained by surrogate-gradient backpropagation through time.
Per-layer thresholds are calibrated once at initialization to a target firing rate.
Three families are compared at matched parameter count: homogeneous, vertical
(one neuron type per layer) and horizontal (typed parallel paths fused before the
read-out). Seeds are 42, 123 and 999 for the ablation and bake-offs; significance
uses a paired bootstrap with 2000 resamples, and a difference is called consistent
only when its sign holds within every seed.

## Data

- **DIAT-µSAT** (radar, 6 classes, 64×64 read as 64 frames of 64 Doppler bins).
  The scripts read it through a small dataset registry (`datasets.json`) that points
  to a folder holding `X.npy` and `y.npy`. The set is publicly available from IEEE
  DataPort (Kumawat et al., 2022, doi:10.21227/1x2q-8v62). Register the path on first
  run or edit the registry block at the top of the radar scripts.
- **Spiking Heidelberg Digits** (audio, 20 classes). `07_shd_disjoint.py` downloads
  the official speaker-disjoint HDF5 from zenkelab.org and bins each recording to 12
  frames; no manual setup is needed. The mixed-split bake-off (`05`) reads a cached
  binned tensor.

## Running

```
pip install -r requirements.txt
python experiments/02_diat_ablation.py      # headline ablation (Figure 3)
python experiments/07_shd_disjoint.py       # official audio ranking (Table 3)
```

A CUDA GPU is recommended. The scripts fall back to CPU and detect Colab automatically.

## Locked results in this repository

`results/shd_disjoint_summary.json` (official speaker-disjoint SHD, three seeds) gives
the audio ranking of Table 3: Dual-tau-LIF 0.671, CrossInhib-LIF 0.635 and
Doppler-LIF 0.451 at sixth, with the CrossInhib-minus-Doppler audio gap at 0.184 and a
consistent sign across all seeds. The Spearman correlation between the speaker-mixed
and speaker-disjoint orderings is 0.76, matching Figure 4.

`results/cnn_frontier_summary.json` gives the tuned single-neuron references of
Table 2: Doppler-LIF 0.924 (475k parameters), Vanilla-LIF 0.857 and Parametric-LIF
0.852, against a compact convolutional reference at 0.932 (about 24k parameters). The
paper makes no energy claim; the operation-count fields are a proxy only.

## Citation

See `CITATION.cff`. Please cite both the paper and the `dikmen-spiking-neurons` library.

## License

MIT, see `LICENSE`.
