from dataclasses import dataclass, fields, asdict
import json
from argparse import ArgumentParser
from utils.enums import CurriculumSetting, ModularSetting, ModelType, Task

@dataclass
class ModArgs:
    batch_size: int = 64
    data_seed: int = 0
    rule_seed: int = 10
    n_op: int = 2
    hidden_dim: int = 128
    enc_dim: int = 128
    seq_len: int = 10
    x_dim: int = 32
    iterations: int = 50000
    setting: ModularSetting = ModularSetting.ModDich
    type: Task = Task.Regression
    model: ModelType = ModelType.LSTM
    val_freq: int = 10
    patience: int = -1
    save_dir: str = None
    lr: float = 1e-3
    curriculum_setting: CurriculumSetting = CurriculumSetting.Baseline
    iters_per_task: int = -1
    no_log: bool = False

    def __str__(self):
        return json.dumps(asdict(self), indent=2)

    def __post_init__(self):
        if self.iters_per_task == -1:
            self.iters_per_task = self.iterations // self.n_op

def parse_args() -> ModArgs:
    parser = ArgumentParser()
    for field in fields(ModArgs):
        name = field.name
        default = field.default
        name = f'--{name.replace("_", "-")}'
        if field.type == bool:
            parser.add_argument(name, action='store_true')
        else:
            parser.add_argument(name, default=default, type=field.type)
    args = parser.parse_args()
    return ModArgs(**vars(args))