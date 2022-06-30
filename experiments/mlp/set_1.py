from train import run
from utils.args import ModArgs
from utils.enums import ModelType, CurriculumSetting
from utils.utils import dict_product
import wandb
import itertools
import time

train_combinations = {
    "data_seed": [0, 1, 2],
    "rule_seed": [10, 11, 12],
    "type": ['regression', 'classification'],
    'setting': ['mod', 'mod_dich'],
    'n_op': [2, 4, 8, 16, 32],
    'curriculum_setting': [CurriculumSetting.Baseline, CurriculumSetting.Incremental, CurriculumSetting.Sequential]
}

base_args = {
    "batch_size": 64,
    "hidden_dim": 64,
    "enc_dim": 32,
    "iterations": 20000,
    "val_freq": 10,
    "patience": -1,  # disable early stopping
    "save_dir": "mlp/set1",
    "lr": 1e-4,
    "no_log": True,  # don't print val accuracies to save time,
    "x_dim": 1,
    "model": ModelType.MLP
}


def runner():
    for train_args in dict_product(train_combinations):
        args = train_args | base_args
        args = ModArgs(**args)
        # wandb.init(project="modular_curriculum",
        #         entity="omar-s", config=args)
        start_t = time.time()
        run(args)
        print(f'Run time: {time.time() - start_t}')
        # wandb.finish()

exp_start_t = time.time()
runner()
print(f'Total run time: {time.time() - exp_start_t}')