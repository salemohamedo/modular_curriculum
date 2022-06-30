from train import run
from utils.args import ModArgs
import wandb
import time

SAVE_DIR = 'set1_mlp_lr'
seeds = [0, 1, 2]
types = ['regression', 'classification']
settings = ['mod', 'mod_dich']
rules = [2, 4, 8, 16]
args = ModArgs(
    batch_size=64,
    hidden_dim=64,
    enc_dim=32,
    iterations=50000,
    val_freq=10,
    patience=-1,
    save_dir=SAVE_DIR,
    model='mlp',
    lr=1e-4)


def runner():
    for type in types:
        for setting in settings:
            for n_op in rules:
                for seed in seeds:
                    args.type = type
                    args.setting = setting
                    args.seed = seed
                    args.n_op = n_op
                    wandb.init(project="modular_curriculum",
                            entity="omar-s", config=args)
                    run_start_t = time.time()
                    run(args)
                    print(f'Run elapsed: {time.time() - run_start_t}')
                    wandb.finish()


# if __name__ == '__main__':
start = time.time()
runner()
print(f'Total elapsed: {time.time() - start}')