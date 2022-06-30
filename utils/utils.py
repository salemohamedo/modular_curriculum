import torch
import random
import numpy as np
import itertools
import pandas as pd
from utils.args import ModArgs
from utils.enums import ModelType
from dataclasses import fields
from functools import reduce

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def load_df(path):
    return pd.read_pickle(path)

def dict_product(dicts):
    return list((dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))

ignore_fields = ["save_dir", "no_log"]

def get_id_fields(args: ModArgs):
    id_fields = [f.name for f in fields(args) if f.name not in ignore_fields]
    id_fields.sort()
    return [(f, getattr(args, f)) for f in id_fields]

def generate_model_id(args: ModArgs):
    id_fields = get_id_fields(args)
    return reduce(lambda x, field: x + f"_{field[0]}_{str(field[1])}",
                  id_fields, "model")