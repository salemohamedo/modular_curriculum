from enum import Enum

class BaseEnum(str, Enum):
    def __str__(self):
        return self.value

class CurriculumSetting(BaseEnum):
    Vanilla = "vanilla"
    Baseline = "baseline"
    Sequential = "sequential"
    Incremental = "incremental"

class ModularSetting(BaseEnum):
    Mod = "mod"
    ModDich = "mod_dich"

class ModelType(BaseEnum):
    MLP = "mlp"
    LSTM = "lstm"

class Task(BaseEnum):
    Regression = "regression"
    Classification = "classification"
