import time
import pickle
import torch
from dataclasses import asdict
from functools import reduce
from model import ModularMLP, ModularLSTM
import pandas as pd
from data import generate_data
from utils.utils import set_seed, generate_model_id
from utils.args import parse_args, ModArgs
from utils.enums import CurriculumSetting, ModelType, Task
from pathlib import Path
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args: ModArgs, model, optim, criterion, allowed_rule_set=None):
    model.train()
    loss = 0.
    x, y, op = generate_data(args=args, allowed_rule_set=allowed_rule_set)
    x, y, op = x.to(device), y.to(device), op.to(device)
    optim.zero_grad()
    out, scores = model(x, op)
    loss = criterion(out, y)
    loss.backward()
    optim.step()
    return float(loss)


@torch.no_grad()
def eval(args: ModArgs, model, criterion, num_evals=10, ood=False, allowed_rule_set=None):
    model.eval()
    total_loss = 0.
    total_acc = 0.
    for i in range(num_evals):
        x, y, op = generate_data(args=args, ood=ood, allowed_rule_set=allowed_rule_set)
        x, y, op = x.to(device), y.to(device), op.to(device)
        out, scores = model(x, op)
        if args.type == Task.Classification:
            acc = torch.eq(out >= 0., y).double().mean()
            total_acc += float(acc)
        loss = criterion(out, y)
        total_loss += float(loss)
    total_loss = total_loss / num_evals
    if args.type == Task.Classification:
        total_acc = total_acc / num_evals
        return {'loss': total_loss, 'acc': total_acc}
    else:
        return {'loss': total_loss}

def curriculum_loop(args: ModArgs, model, optim, criterion, model_dir: Path = None):
    if args.curriculum_setting == CurriculumSetting.Baseline:
        train_loop(args, model, optim, criterion, args.iterations, model_dir)
    elif args.curriculum_setting in [CurriculumSetting.Incremental, CurriculumSetting.Sequential]:
        allowed_rule_set = []
        for i in range(args.n_op):
            if args.curriculum_setting == CurriculumSetting.Sequential:
                allowed_rule_set = [i]
            else: ## Incremental curriculum
                allowed_rule_set.append(i)
            task_model_dir = model_dir / str(i)
            task_model_dir.mkdir()
            train_loop(args, model, optim, criterion,
                       args.iters_per_task, task_model_dir, allowed_rule_set)
    
        ## Evaluate on all primitives
        final_metrics = eval(args, model, criterion, allowed_rule_set=None)
        print(f'Final test loss: {final_metrics["loss"]:.3f}')
    


def train_loop(args: ModArgs, model, optim, criterion, iterations, model_dir: Path = None, allowed_rule_set=None):
    best_val = float('inf')
    bad_iters = 0
    best_model = None
    test_stats = []
    for i in range(iterations):
        if i % args.val_freq == 0:
            val_metrics = eval(args, model, criterion, allowed_rule_set=allowed_rule_set)
            test_metrics = eval(args, model, criterion,
                                allowed_rule_set=allowed_rule_set)
            for m, v in test_metrics.items():
                test_stats.append({'metric': m, 'value': v, 'iteration': i})

        train_loss = train(args, model, optim, criterion, allowed_rule_set=allowed_rule_set)

        if i % args.val_freq == 0:
            log = f'Iteration: {i} | Train Loss: {train_loss}\n' \
                f'Iteration: {i} | Eval Loss: {test_metrics["loss"]}\n'
            if not args.no_log:
                print(log)

            if val_metrics['loss'] < best_val:
                best_val = val_metrics['loss']
                bad_iters = 0
                if args.save_dir is not None:
                    best_model = model.state_dict()
            else:
                bad_iters += 1
                if bad_iters == args.patience:
                    break
    print(f'Best val loss: {best_val:.3f}')
    if args.save_dir is not None:
        torch.save(best_model, model_dir / 'model_best.pt')
        pd.DataFrame(test_stats).to_pickle(model_dir / 'test_stats.pk')


def run(args: ModArgs):
    print(args)
    model_id = generate_model_id(args)
    model_dir = None
    if args.save_dir is not None:
        model_dir = Path('saved_models') / args.save_dir / model_id
        if model_dir.exists():
            print(f'Model already exists at {model_dir}. Exiting...')
            return
        model_dir.mkdir(parents=True)
        with open(model_dir / 'args.pk', 'wb') as f:
            pickle.dump(args, f)
    set_seed(args.data_seed)

    if args.model == ModelType.MLP:
        model = ModularMLP.from_args(args).to(device)
    elif args.model == ModelType.LSTM:
        model = ModularLSTM.from_args(args).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.type == Task.Regression:
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    curriculum_loop(args, model, optim, criterion, model_dir)


if __name__ == '__main__':
    args = parse_args()
    # args = ModArgs(
    #     iterations=200, 
    #     curriculum_setting=CurriculumSetting.Sequential,
    #     model=ModelType.MLP,
    #     n_op=10,
    #     iters_per_task=20,
    #     save_dir="test",
    #     x_dim=1)
    run(args)

