import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import ModArgs
import math

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.network(x.float())

class VanillaRNN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, device):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.i2h = nn.Linear(in_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        h = torch.zeros(self.hidden_size, device=self.device)
        out = []
        for i in range(x.shape[1]):
            h = torch.tanh(self.i2h(x[:,i,:]) + self.h2h(h))
            out.append(self.h2o(h))
        return torch.stack(out, dim=1)


class GroupRNNCell(nn.Module):
    def __init__(self, mod_in_size, router_in_size, hidden_size, n_modules):
        super(GroupRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.n_modules = n_modules
        self.i2h = nn.Linear(mod_in_size, hidden_size * n_modules)
        self.h2h = nn.Linear(hidden_size, hidden_size * n_modules)
    
        self.module_router = nn.Sequential(
            nn.Linear(router_in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_modules),
            nn.Softmax(dim=0)
        )

    def forward(self, x, h, op=None):
        batch_size = x.size(0)
        h = torch.tanh(self.i2h(x) + self.h2h(h))
        h = h.view((batch_size, self.n_modules, -1))
        if op is not None: ## mod_dich setting
            module_score_weights = self.module_router(op)
        else:              ## mod setting
            module_score_weights = self.module_router(x)
        ## module_score_weights: batch_size x n_modules
        # h: batch_size x n_modules x hidden_size
        ## return batch_size x hidden size 
        return torch.einsum('bnh, bn->bh', (h, module_score_weights))

class ModularRNN(nn.Module):
    def __init__(self, in_size, out_size, enc_out_size, hidden_size, n_modules, num_ops, setting):
        super(ModularRNN, self).__init__()
        self.hidden_size = hidden_size
        self.setting = setting

        if setting == 'mod':
            enc_in_size = in_size + num_ops
            router_in_size = enc_out_size
        elif setting == 'mod_dich':
            enc_in_size = in_size
            router_in_size = num_ops

        self.encoder = nn.Sequential(
            nn.Linear(enc_in_size, enc_out_size),
            nn.ReLU(),
            nn.Linear(enc_out_size, enc_out_size)
        )

        self.rnn = GroupRNNCell(
            enc_out_size, router_in_size, hidden_size, n_modules)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x, op):
        x = self.encoder(x)
        batch_size = x.size(0)
        seq_len = x.size(1)
        h = torch.zeros((batch_size, self.hidden_size), device=x.device)
        hs = []
        for i in range(seq_len):
            if self.setting == 'mod':
                h = self.rnn(x[:, i, :], h, None)
            elif self.setting == 'mod_dich':
                h = self.rnn(x[:, i, :], h, op[:, i, :])
            hs.append(h)
        hs = torch.stack(hs, dim=1)
        return self.decoder(hs)
    
    @classmethod
    def from_args(cls, args: ModArgs):
        return cls(
            in_size = args.x_dim,
            out_size = args.x_dim if args.type == 'regression' else 1,
            enc_out_size = args.enc_dim,
            hidden_size = args.hidden_dim,
            n_modules = args.n_op,
            num_ops = args.n_op,
            setting = args.setting
        )

"""
Model below taken from: https://github.com/sarthmit/Mod_Arch
@misc{https://doi.org/10.48550/arxiv.2206.02713,
  doi = {10.48550/ARXIV.2206.02713},
  url = {https://arxiv.org/abs/2206.02713},
  author = {Mittal, Sarthak and Bengio, Yoshua and Lajoie, Guillaume},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Is a Modular Architecture Enough?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

class GroupLinearLayer(nn.Module):
    """Modularized Linear Layer"""

    def __init__(self, num_blocks, din, dout, bias=True):
        super(GroupLinearLayer, self).__init__()

        self.bias = bias
        self.num_blocks = num_blocks
        self.din = din
        self.dout = dout
        self.w = nn.Parameter(torch.Tensor(num_blocks, din, dout))
        self.b = nn.Parameter(torch.Tensor(1, num_blocks, dout))

        self.reset_parameters()

    def reset_parameters(self):
        bound = math.sqrt(1.0 / self.din)
        nn.init.uniform_(self.w, -bound, bound)
        if self.bias:
            nn.init.uniform_(self.b, -bound, bound)

    def extra_repr(self):
        return 'groups={}, in_features={}, out_features={}, bias={}'.format(
            self.num_blocks, self.din, self.dout, self.bias is not None
        )

    def forward(self, x):
        # x - (bsz, num_blocks, din)
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.w)
        x = x.permute(1, 0, 2)

        if self.bias:
            x = x + self.b

        return x


"""
Model below taken from: https://github.com/sarthmit/Mod_Arch
@misc{https://doi.org/10.48550/arxiv.2206.02713,
  doi = {10.48550/ARXIV.2206.02713},
  url = {https://arxiv.org/abs/2206.02713},
  author = {Mittal, Sarthak and Bengio, Yoshua and Lajoie, Guillaume},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Is a Modular Architecture Enough?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

class GroupLSTMCell(nn.Module):
    """
    GroupLSTMCell can compute the operation of N LSTM Cells at once.
    """

    def __init__(self, inp_size, hidden_size, num_lstms, gt=False, op=False, bias=True):
        super(GroupLSTMCell, self).__init__()

        self.num_lstms = num_lstms
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.op = op
        self.gt = gt

        self.i2h = nn.Linear(inp_size, 4 * num_lstms * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * num_lstms *
                             hidden_size, bias=bias)

        if self.gt:
            pass
        elif self.op:
            self.scorer = nn.Sequential(
                nn.Linear(num_lstms, hidden_size, bias=bias),
                nn.ReLU(),
                nn.Linear(hidden_size, num_lstms, bias=bias)
            )
        else:
            self.scorer = nn.Sequential(
                nn.ReLU(),
                GroupLinearLayer(num_lstms, 8 * hidden_size, 1, bias=bias)
            )

    def forward(self, x, hid_state, op=None):
        """
        input: x (batch_size, input_size)
               hid_state (tuple of length 2 with each element of size (batch_size, num_lstms, hidden_state))
        output: h (batch_size, hidden_state)
                c ((batch_size, hidden_state))
        """
        h, c = hid_state
        bsz = h.shape[0]

        i_h = self.i2h(x).reshape(bsz, self.num_lstms, 4 * self.hidden_size)
        h_h = self.h2h(h).reshape(bsz, self.num_lstms, 4 * self.hidden_size)

        if self.gt:
            score = op.unsqueeze(-1)
        elif self.op:
            score = F.softmax(self.scorer(op), dim=1).unsqueeze(-1)
        else:
            score = F.softmax(self.scorer(
                torch.cat((i_h, h_h), dim=-1)), dim=1)

        preact = i_h + h_h

        gates = preact[:, :, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, :, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :, :self.hidden_size]
        f_t = gates[:, :, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, :, -self.hidden_size:]

        c_t = torch.mul(c.unsqueeze(1), f_t) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, c_t.tanh())

        h_t = (h_t * score).sum(dim=1)
        c_t = (c_t * score).sum(dim=1)

        return h_t, c_t, score


"""
Model below taken from: https://github.com/sarthmit/Mod_Arch
@misc{https://doi.org/10.48550/arxiv.2206.02713,
  doi = {10.48550/ARXIV.2206.02713},
  url = {https://arxiv.org/abs/2206.02713},
  author = {Mittal, Sarthak and Bengio, Yoshua and Lajoie, Guillaume},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Is a Modular Architecture Enough?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

class ModularLSTM(nn.Module):
    def __init__(self, in_size, out_size, enc_out_size, hidden_size, n_modules, num_ops, setting):
        super(ModularLSTM, self).__init__()

        if setting == 'mod':
            in_size += n_modules
        self.in_dim = in_size
        self.enc_dim = enc_out_size
        self.hid_dim = hidden_size
        self.out_dim = out_size
        self.num_rules = n_modules
        self.model = setting
        self.op = setting == 'mod_dich'

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.enc_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.enc_dim, self.enc_dim, bias=True)
        )

        hid_new = (4 * self.num_rules * self.enc_dim + 1) ** 2
        hid_new += 4 * (4 * self.num_rules + 1) * (5 * self.hid_dim *
                                                   self.hid_dim + self.hid_dim * (4 * self.enc_dim + 1))
        hid_new = math.sqrt(hid_new) - (4 * self.num_rules * self.enc_dim + 1)
        hid_new /= 2 * (4 * self.num_rules + 1)

    #     if self.model == 'Monolithic':
    #         self.rnn = nn.LSTMCell(self.enc_dim, self.hid_dim, bias=bias)
    # # elif self.model == 'Modular':
        self.hid_dim = int(hid_new)
        self.rnn = GroupLSTMCell(
            self.enc_dim, self.hid_dim, self.num_rules, False, self.op, bias=False)
        # elif self.model == 'GT_Modular':
        #     self.hid_dim = int(hid_new)
        #     self.rnn = GroupLSTMCell(
        #         self.enc_dim, self.hid_dim, self.num_rules, True, False, bias=False)
        # else:
        #     print("No Algorithm")
        #     exit()

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim, self.hid_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim, bias=True)
        )

    def forward(self, x, op):
        # x - (bsz, time, dim)
        # op - (bsz, time, rules)

        x = self.encoder(x)

        h = torch.zeros([x.shape[0], self.hid_dim]).cuda()
        c = torch.zeros_like(h)

        out = []
        scores = []

        for i in range(x.shape[1]):
            # if self.model == 'Monolithic':
            #     h, c = self.rnn(x[:, i, :], (h, c))
            # else:
            h, c, score = self.rnn(x[:, i, :], (h, c), op[:, i, :])
            scores.append(score)

            out.append(h)

        out = torch.stack(out)
        # if self.model == 'Monolithic':
        #     scores = None
        # else:
        scores = torch.stack(scores).transpose(1, 0).contiguous()

        out = self.decoder(out).transpose(1, 0).contiguous()

        # return out
        return out, scores

    @classmethod
    def from_args(cls, args: ModArgs):
        return cls(
            in_size=args.x_dim,
            out_size=args.x_dim if args.type == 'regression' else 1,
            enc_out_size=args.enc_dim,
            hidden_size=args.hidden_dim,
            n_modules=args.n_op,
            num_ops=args.n_op,
            setting=args.setting
        )

"""
Model below taken from: https://github.com/sarthmit/Mod_Arch
@misc{https://doi.org/10.48550/arxiv.2206.02713,
  doi = {10.48550/ARXIV.2206.02713},
  url = {https://arxiv.org/abs/2206.02713},
  author = {Mittal, Sarthak and Bengio, Yoshua and Lajoie, Guillaume},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Is a Modular Architecture Enough?},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""

class ModularMLP(nn.Module):
    def __init__(self, op_dim, encoder_dim, dim, num_rules, setting):
        super(ModularMLP, self).__init__()

        self.dim = dim
        self.encoder_dim = encoder_dim
        self.op_dim = op_dim
        self.num_rules = num_rules
        self.setting = setting

        self.encoder_digit = nn.Sequential(
            nn.Linear(1, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        self.encoder_operation = nn.Sequential(
            nn.Linear(op_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, encoder_dim)
        )

        if setting == "mod":
            modular_in_size = encoder_dim * 3 ## module gets both digits + task
        elif setting == "mod_dich":
            modular_in_size = encoder_dim * 2 ## module only gets digits
        self.MLP = nn.Sequential(
            GroupLinearLayer(num_rules, modular_in_size, dim // num_rules),
            nn.ReLU(),
            GroupLinearLayer(num_rules, dim // num_rules, dim)
        )
        if setting == "mod_dich":
            self.scorer = nn.Linear(encoder_dim, num_rules)
        elif setting == "mod":
            self.scorer = nn.Sequential(
                nn.Linear(dim + op_dim + 2, encoder_dim),
                nn.ReLU(),
                nn.Linear(encoder_dim, 1)
            )

        self.decoder = nn.Sequential(
            nn.Linear(dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 1)
        )

    def forward(self, x, op):
        if self.setting == "mod":
            dig1, dig2, op = x[:, 0:1], x[:, 1:2], x[:, 2:]
        elif self.setting == "mod_dich":
            assert op is not None
            dig1, dig2 = x[:, 0:1], x[:, 1:]

        dig1 = self.encoder_digit(dig1)
        dig2 = self.encoder_digit(dig2)
        op = self.encoder_operation(op)

        if self.setting == "mod":
            sample = torch.cat((dig1, dig2, op), dim=-1)
        elif self.setting == "mod_dich":
            sample = torch.cat((dig1, dig2), dim=-1)

        sample = sample.unsqueeze(1).repeat(1, self.num_rules, 1)


        out = self.MLP(sample)
        if self.setting == "mod_dich":
            score = F.softmax(self.scorer(op), dim=-1)
            score = score.unsqueeze(-1)
        elif self.setting == "mod":
            score = torch.cat(
                [out, x.unsqueeze(1).repeat(1, self.num_rules, 1)], dim=-1)
            score = F.softmax(self.scorer(score), dim=1)

        out = (out * score).sum(dim=1)

        return self.decoder(out).squeeze(), score.squeeze()

    @classmethod
    def from_args(cls, args: ModArgs):
        return cls(
            op_dim=args.n_op,
            encoder_dim=args.enc_dim,
            dim=args.hidden_dim,
            num_rules=args.n_op,
            setting=args.setting
        )
