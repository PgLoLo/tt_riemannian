from typing import Sequence

import torch as t
from torch import nn

from dataset_reduction.dl_routine import on_cpu, device, list_to_device
from dataset_reduction.score.tt.core.tt_network import TTNetwork
from dataset_reduction.score.tt.core.orthogonalization import prepare_for_tangent, deltas_to_tangent


def prepare_for_AD_in_tangent(tt: TTNetwork):
    with on_cpu(tt), t.no_grad():
        left, right, s = prepare_for_tangent(tt)

    d = device(tt)
    left = list_to_device(left, d)
    right = list_to_device(right, d)
    s = list_to_device(s, d)

    rs = nn.ParameterList([nn.Parameter(s[0])] + [nn.Parameter(t.zeros_like(r)) for r in s[1:]])
    tt_tangent = deltas_to_tangent(left, right, rs)

    return rs, tt_tangent, left, right, s


def apply_gauge_on_grad(left: Sequence[t.Tensor], rs: Sequence[t.Tensor]):
    fixed_grads = []
    for r, l in zip(rs[:-1], left[:-1]):
        D = r.grad.reshape(-1, r.shape[2])
        U = l.reshape(-1, l.shape[2])
        D = D + t.matmul(U, t.matmul(U.T, D))
        fixed_grads.append(D.reshape(r.shape))
    fixed_grads.append(rs[-1].grad)

    return fixed_grads