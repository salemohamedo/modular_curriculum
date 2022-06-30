import torch
from torch.utils.data import TensorDataset
import numpy as np
import math

from utils.args import ModArgs
from utils.enums import ModularSetting, ModelType, Task


def rnn_generate_data(n, seq_len, num_rules, dim, rule_seed, setting, type=Task.Regression, test_mode=False, ood=False, p=None, allowed_rule_set=None):
    rng = torch.Generator().manual_seed(rule_seed)
    co_a = torch.randn((num_rules, dim, dim), generator=rng) / math.sqrt(dim)
    co_b = torch.randn((num_rules, dim, dim), generator=rng) / math.sqrt(dim)

    if allowed_rule_set is None:
        allowed_rule_set = np.arange(num_rules)
    if p is None:
        task_ids = np.random.choice(list(allowed_rule_set), size=(n, seq_len), p=None)
    else:
        task_ids = np.random.choice(list(allowed_rule_set), size=(n, seq_len), p=p)
    
    task_ids = torch.from_numpy(task_ids)

    x = torch.randn(n, seq_len, dim)
    if ood:
        x = x * 2
    task_onehot = torch.nn.functional.one_hot(task_ids, num_rules).float()
    y = torch.zeros_like(x)
    prev_state = torch.zeros((n, dim))

    for i in range(seq_len):
        co_a_i = torch.index_select(co_a, dim=0, index=task_ids[:, i])
        co_b_i = torch.index_select(co_b, dim=0, index=task_ids[:, i])
        y[:, i, :] = torch.einsum(
            'bnh, bn->bh', (co_a_i, prev_state)) + torch.einsum(
            'bnh, bn->bh', (co_b_i, x[:, i, :]))
        prev_state = y[:, i, :]

    if setting == ModularSetting.Mod:
        x = torch.concat([x, task_onehot], dim=-1)

    dec = None
    if type == Task.Classification:
        # dec = rng.randn(dim, 1)
        dec = torch.ones((dim, 1))
        y = torch.matmul(y, dec) >= 0.

    if test_mode:
        return x, y.float(), task_onehot, co_a, co_b, dec
    else:
        return x, y.float(), task_onehot


def mlp_generate_data(n, num_rules, rule_seed, ood=None, prob=None, type='regression', setting='mod_dich', test_mode=False, allowed_rule_set=None):
    x1 = torch.randn(n)
    x2 = torch.randn(n)

    rng = torch.Generator().manual_seed(rule_seed)
    co_a = torch.randn(num_rules, generator=rng)
    co_b = torch.randn(num_rules, generator=rng)

    if allowed_rule_set is None:
        allowed_rule_set = np.arange(num_rules)
    if prob is None:
        task_ids = np.random.choice(
            list(allowed_rule_set), size=(n,), p=None)
    else:
        task_ids = np.random.choice(
            list(allowed_rule_set), size=(n,), p=prob)
    task_ids = torch.from_numpy(task_ids)

    y = x1.mul(co_a[task_ids]) + x2.mul(co_b[task_ids])
    x1, x2 = x1.unsqueeze(-1), x2.unsqueeze(-1)
    if type == Task.Classification:
        y = y >= 0.
        y = y.float()
    
    task_onehot = torch.nn.functional.one_hot(task_ids, num_rules).float()
    if setting == ModularSetting.ModDich:
        x = torch.concat([x1, x2], dim=-1)
    elif setting == ModularSetting.Mod:
        x = torch.concat([x1, x2, task_onehot], dim=-1)
    
    if test_mode:
        return x, y, task_onehot, co_a, co_b
    else:
        return x, y, task_onehot

def generate_data(args: ModArgs, ood=False, prob=None, n=None, allowed_rule_set=None):
    n = n if n is not None else args.batch_size
    if args.model == ModelType.LSTM:
        return rnn_generate_data(
            n=n,
            seq_len=args.seq_len,
            num_rules=args.n_op,
            dim=args.x_dim,
            rule_seed=args.rule_seed,
            type=args.type,
            setting=args.setting,
            ood=ood,
            p=prob,
            allowed_rule_set=allowed_rule_set)
    elif args.model == ModelType.MLP:
        return mlp_generate_data(
            n=n,
            num_rules=args.n_op,
            rule_seed=args.rule_seed,
            type=args.type,
            setting=args.setting,
            ood=ood,
            prob=prob,
            allowed_rule_set=allowed_rule_set)


def _verify_mlp_data_gen(x, y, task_ids, co_a, co_b, classify=False):
    task_ids = torch.argmax(task_ids, dim=-1)
    n = x.shape[0]
    for i in range(n):
        y_cal = torch.mul(co_a[task_ids[i]], x[i][0]) + \
            torch.mul(co_b[task_ids[i]], x[i][1])
        if classify:
            y_cal = y_cal >= 0.
            y_cal = y_cal.float()
        assert y_cal == y[i]

def _verify_rnn_data_gen(x, y, task_ids, co_a, co_b, dec, classify=False):
    task_ids = torch.argmax(task_ids, dim=-1)
    n = x.shape[0]
    seq_len = x.shape[1]
    x_dim = x.shape[2]
    for i in range(n):
        prev_state = torch.zeros(x_dim)
        for l in range(seq_len):
            task = task_ids[i][l]
            y_calc = torch.matmul(prev_state, co_a[task]) + torch.matmul(x[i][l], co_b[task])
            if classify:
                y_calc_class = torch.dot(y_calc, dec.squeeze()) >= 0.
                y_calc_class = y_calc_class.float()
                assert torch.equal(y_calc_class, y[i][l].squeeze())
            else:
                assert torch.allclose(y_calc, y[i][l], atol=1e-4)
            prev_state = y_calc

if __name__ == '__main__':
    from utils.utils import set_seed
    set_seed(0)
    for i in range(5):
        x, y, task_ids, co_a, co_b, dec = rnn_generate_data(n=100, seq_len=30, num_rules=300, dim=30, rule_seed=i, setting='nomod', type='regression', test_mode=True)
        _verify_rnn_data_gen(x, y, task_ids, co_a, co_b, dec, classify=False)
        x, y, task_ids, co_a, co_b, dec = rnn_generate_data(
            n=100, seq_len=30, num_rules=300, dim=30, rule_seed=i, setting='mod_dich', type='classification', test_mode=True)
        _verify_rnn_data_gen(x, y, task_ids, co_a, co_b, dec, classify=True)
        
        x, y, task_ids, co_a, co_b = mlp_generate_data(
            n=100, num_rules=300, rule_seed=i, setting='mod_dich', type='regression', test_mode=True)
        _verify_mlp_data_gen(x, y, task_ids, co_a, co_b, classify=False)
        x, y, task_ids, co_a, co_b = mlp_generate_data(
            n=100, num_rules=300, rule_seed=i, setting='mod_dich', type='classification', test_mode=True)
        _verify_mlp_data_gen(x, y, task_ids, co_a, co_b, classify=True)

    x, y, task_ids, co_a, co_b, dec = rnn_generate_data(
        n=3, seq_len=5, num_rules=20, dim=3, rule_seed=0, setting='mod_dich', type='regression', test_mode=True, allowed_rule_set=[3, 14, 9])
    assert [3, 9, 14] == sorted(torch.argmax(task_ids, dim=-1).unique().tolist())

    x, y, task_ids, co_a, co_b = mlp_generate_data(
        n=100, num_rules=300, rule_seed=0, setting='mod_dich', type='classification', test_mode=True, allowed_rule_set=[89, 23])
    assert [23, 89] == sorted(torch.argmax(task_ids, dim=-1).unique().tolist())
