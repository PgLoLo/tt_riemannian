import torch as t

from dataset_reduction.score.tt.core.tt_network import TTNetwork
from dataset_reduction.score.tt.core.orthogonalization import get_orthogonalized


def round_tt(tt: TTNetwork, new_rang: int) -> TTNetwork:
    tt = get_orthogonalized(tt, right_to_left=True)

    G_new = []

    remainder = None
    for i, G in enumerate(tt.Gs[:-1]):
        if remainder is not None:
            G = t.einsum('il,ljk->ijk', remainder, G)
        U, S, V = t.svd(G.reshape(-1, G.shape[2]))

        G_new.append(U[:, :new_rang].reshape(*G.shape[:2], -1))
        remainder = S[:new_rang, None] * V.T[:new_rang]

    G_new.append(t.einsum('il,ljk->ijk', remainder, tt.Gs[-1]))

    return TTNetwork(G_new)
