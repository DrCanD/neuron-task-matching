# Locked result summaries

These JSON files are the verified run summaries that back the paper tables. The
per-seed scores are kept so the means, standard deviations and paired-bootstrap
statements can be re-derived without re-training.

- `shd_disjoint_summary.json` — SHD audio, official speaker-disjoint split, three
  seeds. Backs Table 3 (per-neuron ranking, Doppler-LIF sixth) and the Spearman
  robustness check of Figure 4 (mixed vs disjoint, rho = 0.76). Also records the
  CrossInhib-minus-Doppler audio gap (0.184, consistent sign across seeds).

- `cnn_frontier_summary.json` — DIAT-µSAT tuned best models and the compact-CNN
  accuracy-cost frontier. Backs Table 2 (Doppler-LIF 0.924 / 475k, Vanilla 0.857,
  Parametric 0.852, CNN reference 0.932 / ~24k). The energy fields are an
  operation-count proxy and are not an energy claim.

Full per-seed, per-neuron result files (one JSON per neuron and seed) are produced by
the scripts into their run folders.
