from train import run
from utils.args import CurriculumSetting, ModArgs
import wandb
import itertools


def dict_product(dicts):
    return list((dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))


train_combinations = {
    "data_seed": [0, 1, 2],
    "rule_seed": [10, 11, 12],
    "type": ['regression', 'classification'],
    'setting': ['mod', 'mod_dich'],
    'n_op': [2, 4, 8, 16, 32, 64],
    'curriculum_setting': [CurriculumSetting.Vanilla, CurriculumSetting.Sequential, CurriculumSetting.Incremental]
}

base_args = {
    "batch_size": 64,
    "hidden_dim": 128,
    "enc_dim": 128,
    "iterations": 10000,
    "val_freq": 10,
    "patience": -1,  # disable early stopping
    "save_dir": "curriculum_set1"
}


def runner():
    for train_args in dict_product(train_combinations):
        args = train_args | base_args
        args = ModArgs(**args)
        # wandb.init(project="modular_curriculum",
        #         entity="omar-s", config=args)
        run(args)
        # wandb.finish()


runner()
