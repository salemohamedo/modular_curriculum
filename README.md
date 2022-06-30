# Curriculum Learning with Modular Architectures
### **Note: this repo is still a WIP**
## Main files:
* `train.py`: train an MLP/LSTM model on a particular curricular/modular setup. Model checkpoints can also be saved for evaluation later.
* `evaluate_{lstm/mlp}.py`: evaluate performance/specialization of trained models
* `data.py`: Implements data generation for MLP/LSTM models
* `model.py`: defines both an MLP and LSTM modular architecture, adapted from [Mittal et al (2022)](https://github.com/sarthmit/Mod_Arch)
* `utils/args.py`: Defines the `ModArgs` class which describes the different settings supported in `train.py`
* `experiments/`: Each file in this folder defines a specific set of experiments. It is used as a runner to train various models (via `train.py`). To invoke an experiment (e.g. `lstm/set_1`), run: `python -m experiments/lstm/set_1`