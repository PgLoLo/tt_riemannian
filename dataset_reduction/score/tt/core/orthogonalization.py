from typing import Sequence

import torch as t

from dataset_reduction.score.tt.core.tt_network import TTNetwork


__all__ = ['orthogonalize', 'left_to_right_core', 'prepare_for_tangent']

from dataset_reduction.score.tt.utils import lists_to_rang2_concat, reverse_and_transpose


def orthogonalize_core(G: t.Tensor):
    assert G.dim() == 3
    q, r = t.qr(G.reshape(-1, G.shape[2]))
    return q.reshape(G.shape[0], G.shape[1], -1), r


def orthogonalize(tt: TTNetwork, *, right_to_left: bool = False):
    if right_to_left:
        tt = tt.construct_reverse()

    qs = []
    rs = []

    r = None
    for G in tt.Gs:
        if r is not None:
            G = t.einsum('ij,jkl->ikl', r, G)

        q, r = orthogonalize_core(G)
        qs.append(q)
        rs.append(r)

    if right_to_left:
        qs = reverse_and_transpose(qs)
        rs = reverse_and_transpose(rs)

    return qs, rs


def get_orthogonalized(tt: TTNetwork, right_to_left: bool = False) -> TTNetwork:
    if right_to_left:
        return get_orthogonalized(tt.construct_reverse()).construct_reverse()

    qs, rs = orthogonalize(tt)
    qs[-1] = t.einsum('ijl,lk->ijk', qs[-1], rs[-1])

    return TTNetwork(qs)


def prepare_for_tangent(tt: TTNetwork):
    assert tt.n_dims > 1

    left, s_left = orthogonalize(tt)
    right, s_right = orthogonalize(tt, right_to_left=True)
    s = []
    for i, G in enumerate(tt.Gs):
        if i == 0:
            s.append(t.einsum('ijk,kl->ijl', G, s_right[i + 1]))
        elif i == tt.n_dims - 1:
            s.append(t.einsum('ij,jkl->ikl', s_left[i - 1], G))
        else:
            s.append(t.einsum('ij,jkl,ln->ikn', s_left[i - 1], G, s_right[i + 1]))

    return left, right, s


def deltas_to_tangent(left: Sequence[t.Tensor], right: Sequence[t.Tensor], s: Sequence[t.Tensor]) -> TTNetwork:
    assert len(left) > 1

    Gs = lists_to_rang2_concat(main=s, upper=right, lower=left, row_axis=0, col_axis=2)
    return TTNetwork(Gs)


def left_to_right_core(G_from: t.Tensor, G_to: t.Tensor):
    q, r = orthogonalize_core(G_from)
    G_from.data.copy_(q)
    G_to.data.copy_(t.einsum('ij,jkl->ikl', r, G_to))
    return q, r


def right_to_left_core(G_from: t.Tensor, G_to: t.Tensor):
    q, r = orthogonalize_core(G_from.permute((2, 1, 0)))
    q = q.permute((2, 1, 0))
    r = r.T
    G_from.data.copy_(q)
    G_to.data.copy_(t.einsum('ijk,kl->ijl', G_to, r))
    return q, r
