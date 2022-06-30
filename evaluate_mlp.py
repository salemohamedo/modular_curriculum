import pandas as pd
import numpy as np
import torch
from model import ModularMLP
from utils.args import ModArgs
import pickle
from pathlib import Path
from typing import Tuple
from train import eval
from dataclasses import asdict
from tqdm import tqdm
from utils.utils import set_seed
from utils.enums import Task
from data import generate_data
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--models-dir', type=str)
parser.add_argument('--results-dir', type=str)
args = parser.parse_args()

SAVED_MODELS_DIR = Path(args.models_dir)
RESULTS_OUT_DIR = Path(args.results_dir)
if not SAVED_MODELS_DIR.exists():
    print(f"{SAVED_MODELS_DIR} does not exist!")
    exit()
if not RESULTS_OUT_DIR.exists():
    RESULTS_OUT_DIR.mkdir(parents=True)

def load_args(model_dir) -> ModArgs:
    with open(model_dir / 'args.pk', 'rb') as f:
        return pickle.load(f)

def load_model_and_args(model_dir) -> Tuple[torch.nn.Module, ModArgs]:
    args = load_args(model_dir)
    set_seed(args.seed)
    model = ModularMLP.from_args(args).to(device)
    model.load_state_dict(torch.load(model_dir / 'model_best.pt'))
    return model, args

def perf_metrics():
    results = []
    for model_dir in tqdm(SAVED_MODELS_DIR.iterdir()):
        model, args = load_model_and_args(model_dir)
        if args.type == Task.Regression:
            criterion = torch.nn.L1Loss()
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
        for dist in ['iid', 'ood']:
            metrics = eval(args, model, criterion, ood=(dist=='ood'))
            id_fields = get_id_fields(args)
            for m in metrics.keys():
                d = {f[0]: f[1] for f in id_fields}
                d['metric'] = m
                d['value'] = metrics[m]
                d['dist'] = dist
                results.append(d)
    pd.DataFrame(results).to_pickle(RESULTS_OUT_DIR / 'eval_results.pk')

def train_metrics():
    dfs = []
    for model_dir in tqdm(SAVED_MODELS_DIR.iterdir()):
        args = load_args(model_dir)
        df = pd.read_pickle(model_dir / 'test_stats.pk')
        id_fields = get_id_fields(args)
        for f in id_fields:
            df[f[0]] = f[1]
        dfs.append(df)
    pd.to_pickle(pd.concat(dfs, ignore_index=True), RESULTS_OUT_DIR / 'train_results.pk')

def specialization_score(true_p, empirical_p):
    true_p.sort()
    empirical_p.sort()
    return np.sum(np.abs(true_p - empirical_p)) / 2.

def prob_metrics(args: ModArgs, model):
    prob = np.zeros([args.n_op, args.n_op])
    total = np.zeros([args.n_op, 1])

    with torch.no_grad():
        for _ in range(1000):
            x, y, op = generate_data(args, ood=False, prob=None, n=1000)
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)
            op = torch.Tensor(op).to(device)

            out, score = model(x, op)
            op = op.view(-1, args.n_op).detach().cpu().numpy()
            score = score.view(-1, args.n_op).detach().cpu().numpy()

            for i in range(args.n_op):
                idx = op[:, i] == 1
                idx = np.reshape(idx, [-1])
                total[i, 0] += np.sum(idx)
                prob[i] += np.sum(score[idx,:], axis=0)

    prob /= total
    prob *= (1./ args.n_op)
    return prob


def collapse_metric_worse(prob, rules):
    p = np.min(np.sum(prob, axis=0))
    cmw = 1 - rules * p
    return cmw


def collapse_metric(prob, rules):
    p = np.sum(prob, axis=0)
    cm = rules * np.sum(np.maximum(np.ones_like(p) /
                        rules - p, 0)) / (rules - 1)
    return cm


def hungarian_metric(prob, rules):
    prob = prob * rules
    cost = 1 - prob
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros((rules, rules))
    perm[row_ind, col_ind] = 1.
    hung_score = np.sum(np.abs(perm - prob)) / (2 * rules)
    return hung_score


def compute_prob_metrics():
    results = []
    for model_dir in tqdm(SAVED_MODELS_DIR.iterdir()):
        model, args = load_model_and_args(model_dir)
        prob = prob_metrics(args, model)
        cps_avg = collapse_metric(prob, args.n_op)
        cps_worst = collapse_metric_worse(prob, args.n_op)
        hungarian = hungarian_metric(prob, args.n_op)
        id_fields = get_id_fields(args)
        d = {f[0]: f[1] for f in id_fields}
        d['collapse_avg'] = cps_avg
        d['collapse_worst'] = cps_worst
        d['hungarian'] = hungarian
        results.append(d)
    pd.DataFrame(results).to_pickle(RESULTS_OUT_DIR / 'prob_results.pk')
        
def specialization_metric():
    results = []
    for model_dir in tqdm(SAVED_MODELS_DIR.iterdir()):
        model, args = load_model_and_args(model_dir)
        spec = 0.
        with torch.no_grad():
            for eval_seed in range(100):
                scores = np.zeros(args.n_op)
                rng = np.random.RandomState(eval_seed)
                p = rng.dirichlet(alpha=np.ones(args.n_op))
                for _ in range(10):
                    x, y, op = x, y, op = generate_data(
                        args, ood=False, prob=p, n=1000)
                    x, op = x.to(device), op.to(device)
                    _, score = model(x, op)
                    scores += score.view(-1, args.n_op).mean(dim=0).detach().cpu().numpy()

                spec += specialization_score(p, scores / 10.)
        id_fields = get_id_fields(args)
        d = {f[0]: f[1] for f in id_fields}
        d['spec'] = spec / 100.
        results.append(d)
    pd.DataFrame(results).to_pickle(RESULTS_OUT_DIR / 'spec_results.pk')

perf_metrics()
train_metrics()
specialization_metric()
compute_prob_metrics()
