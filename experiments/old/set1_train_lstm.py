from train import run
from utils.args import ModArgs
import wandb

seeds = [0, 1, 2]
types = ['regression', 'classification']
settings = ['mod', 'mod_dich']
rules = [2, 4, 8, 16]
# rules = [2, 4, 8, 16, 32]
# ds = [0, 1, 2, 3, 4]
# encs = [32, 64, 128, 256, 512] hidden
# dims = [128, 256, 512, 1024, 2048] encoder 
args = ModArgs(
    batch_size=64,
    hidden_dim=128,
    enc_dim=128,
    iterations=10000,
    val_freq=10,
    patience=-1,
    save=True)

def runner():
    for type in types:
        for setting in settings:
            for n_op in rules:
                for seed in seeds:
                    args.type = type
                    args.setting = setting
                    args.seed = seed
                    args.n_op = n_op
                    # print(args)
                    # wandb.init(project="modular_curriculum",
                    #         entity="omar-s", config=args)
                    # run(args)
                    # wandb.finish()

# if __name__ == '__main__':
runner()
