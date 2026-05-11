# Rhythmic Working Memory

This repository contains the code used to train and analyze a biologically constrained recurrent neural network for studying how rhythmic temporal structure affects sequential working memory.

The model implements an excitatory/inhibitory recurrent neural network with short-term synaptic plasticity (STSP). The task is a delayed match-to-sample sequence-memory task in which the network receives a sequence of stimuli under rhythmic or arrhythmic temporal conditions and then reports whether the test sequence matches the sample sequence.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Repository structure

```text
rhythmic_working_memory/
├── code/
│   ├── main.py                 # Main training entry point
│   ├── test_main.py            # Export hidden activity and synaptic variables from trained models
│   ├── lib/
│   │   ├── EIRNN.py            # E/I recurrent neural network with optional STSP
│   │   ├── dataset.py          # Task generation, stimulus timing, sequence-pair construction
│   │   ├── initialize.py       # Parameter initialization and E/I/STSP setup
│   │   ├── parameters.py       # Default training, dataset, and model parameters
│   │   └── train.py            # Training loop, loss function, accuracy computation, checkpoint saving
│   └── utils/
│       ├── activity_loader.py  # Utilities for loading exported neural and synaptic activity
│       ├── rhythmicity.py      # Rhythmicity, phase, PLV, and power-related analyses
│       ├── selectivity.py      # Selectivity and eta-squared analysis utilities
│       ├── align.py
│       ├── eBOSC.py
│       └── typesetter.py
│
└── figure/
    ├── Fig2/
    ├── Fig3/
    ├── Fig4/
    ├── Fig5/
    └── README.md              # Entry notebooks for manuscript figures
