{
  "run_id": "main",
  "split": "speaker-disjoint (official zenkelab.org)",
  "config": {
    "L": 4,
    "W": 128,
    "norm": "none",
    "target_firing": 0.15,
    "stft_window": 8,
    "T": 12,
    "F": 700,
    "duration_s": 1.0,
    "max_epochs": 200,
    "patience": 20,
    "val_smooth": 5,
    "val_frac": 0.15,
    "select_metric": "macro_f1",
    "lr": 0.001,
    "batch_size": 128,
    "seeds": [
      42,
      123,
      999
    ],
    "nboot": 2000,
    "methods": [
      "CrossInhib-LIF",
      "Vanilla-LIF",
      "Dual-tau-LIF",
      "Chirp-LIF",
      "Beam-IF",
      "Doppler-LIF",
      "STFT-IF",
      "Phase-LIF"
    ],
    "mixed_ranking": {
      "CrossInhib-LIF": 0.741,
      "Vanilla-LIF": 0.703,
      "Dual-tau-LIF": 0.694,
      "Chirp-LIF": 0.662,
      "Beam-IF": 0.639,
      "Doppler-LIF": 0.601,
      "STFT-IF": 0.244,
      "Phase-LIF": 0.143
    },
    "exp_index": 2,
    "exp_total": 3,
    "exp_name": "SHD speaker-disjoint per-neuron ranking",
    "exp_manifest": [
      [
        1,
        "DIAT ablation + dissociation (mixed-split) [paper]",
        "done"
      ],
      [
        2,
        "SHD speaker-DISJOINT ranking re-run",
        "THIS"
      ],
      [
        3,
        "parameter-matched CNN baseline",
        "pending"
      ]
    ]
  },
  "results": {
    "CrossInhib-LIF": {
      "f1_mean": 0.6352155528582272,
      "f1_std": 0.018476316121834042,
      "acc_mean": 0.6464958775029447,
      "f1s": [
        0.6488616060679894,
        0.6090948499684103,
        0.6476902025382818
      ]
    },
    "Vanilla-LIF": {
      "f1_mean": 0.5805751575664683,
      "f1_std": 0.01567573347982745,
      "acc_mean": 0.5906949352179035,
      "f1s": [
        0.5932310253678332,
        0.5900099863627767,
        0.5584844609687946
      ]
    },
    "Dual-tau-LIF": {
      "f1_mean": 0.6708993046859734,
      "f1_std": 0.01282546124244807,
      "acc_mean": 0.677414605418139,
      "f1s": [
        0.679250390163993,
        0.6527798309149192,
        0.6806676929790083
      ]
    },
    "Chirp-LIF": {
      "f1_mean": 0.5811347811383906,
      "f1_std": 0.01793510190831677,
      "acc_mean": 0.5927561837455829,
      "f1s": [
        0.5557855369063328,
        0.5945601917334028,
        0.5930586147754363
      ]
    },
    "Beam-IF": {
      "f1_mean": 0.6249366709003583,
      "f1_std": 0.028065986074944968,
      "acc_mean": 0.6410482921083628,
      "f1s": [
        0.5976954834412666,
        0.6135575118130767,
        0.6635570174467317
      ]
    },
    "Doppler-LIF": {
      "f1_mean": 0.4512091006939434,
      "f1_std": 0.011925172095759426,
      "acc_mean": 0.452002355712603,
      "f1s": [
        0.4360744439895236,
        0.45233273297171417,
        0.46522012512059224
      ]
    },
    "STFT-IF": {
      "f1_mean": 0.09656595642126549,
      "f1_std": 0.018570081308551672,
      "acc_mean": 0.15532979976442873,
      "f1s": [
        0.11707080991462696,
        0.10052391732310324,
        0.07210314202606627
      ]
    },
    "Phase-LIF": {
      "f1_mean": 0.28548696215918057,
      "f1_std": 0.013207069568716236,
      "acc_mean": 0.29372791519434627,
      "f1s": [
        0.26953605776638695,
        0.3018776693533973,
        0.2850471593577574
      ]
    }
  },
  "disjoint_ranking": [
    "Dual-tau-LIF",
    "CrossInhib-LIF",
    "Beam-IF",
    "Chirp-LIF",
    "Vanilla-LIF",
    "Doppler-LIF",
    "Phase-LIF",
    "STFT-IF"
  ],
  "mixed_ranking": [
    "CrossInhib-LIF",
    "Vanilla-LIF",
    "Dual-tau-LIF",
    "Chirp-LIF",
    "Beam-IF",
    "Doppler-LIF",
    "STFT-IF",
    "Phase-LIF"
  ],
  "spearman_mixed_vs_disjoint": 0.7619047619047621,
  "crossinhib_vs_doppler_audio": {
    "pair": "CrossInhib-LIF - Doppler-LIF",
    "mean_diff": 0.18400645216428382,
    "per_seed_ci": [
      [
        0.186,
        0.239
      ],
      [
        0.132,
        0.183
      ],
      [
        0.155,
        0.207
      ]
    ],
    "consistent_sign": true
  },
  "timestamp": "2026-06-18T16:54:12.733794"
}