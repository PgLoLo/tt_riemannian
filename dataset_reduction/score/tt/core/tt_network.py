from typing import Sequence, List, Union

import torch as t
from torch import nn
import numpy as np

from dataset_reduction.dl_routine import squeeze
from dataset_reduction.score.tt.utils import einsum

__all__ = ['TTNetwork', 'tt_dot_product', 'create_param', 'pad_to_rang']


def create_param(*shape: int):
    return nn.Parameter(t.randn(*shape) / np.sqrt(np.prod(shape[:-1])), requires_grad=True)


class TTNetwork(nn.Module):
    @staticmethod
    def randomly_generated(n_dims: int, rang: int, m: int):
        return TTNetwork(nn.ParameterList(
            [create_param((1 if i == 0 else rang), m, (1 if i == n_dims - 1 else rang)) for i in range(n_dims)]
        ))

    def __init__(self, Gs: Union[nn.ParameterList, Sequence[t.Tensor]] = None, to_param_list: bool = False):
        super().__init__()

        if to_param_list:
            Gs = nn.ParameterList([nn.Parameter(G.clone()) for G in Gs])

        for G1, G2 in zip(Gs[:-1], Gs[1:]):
            assert G1.shape[2] == G2.shape[0]

        if isinstance(Gs, nn.ParameterList):
            self.Gs = Gs
        else:
            self.Gs = list(Gs)

    @property
    def n_dims(self):
        return len(self.Gs)

    def to(self, device: t.device = None, dtype: t.dtype = None):
        if isinstance(self.Gs, nn.ParameterList):
            super().to(device=device, dtype=dtype)
        else:
            for i, G in enumerate(self.Gs):
                self.Gs[i] = G.to(device=device, dtype=dtype)
        return self

    def cuda(self, device: t.device = t.device('cuda')):
        return self.to(device)

    def cpu(self):
        return self.to(t.device('cpu'))

    def float(self):
        return self.to(dtype=t.float)

    def double(self):
        return self.to(dtype=t.double)

    def construct_reverse(self):
        return TTNetwork([G.permute(2, 1, 0) for G in self.Gs[::-1]])

    def contract(self, x: t.Tensor) -> List[t.Tensor]:
        return [t.tensordot(x[:, i], G, ((-1,), (1,))) for i, G in enumerate(self.Gs)]

    def forward(self, x: t.Tensor) -> t.Tensor:
        contracted = self.contract(x)
        return self.value(contracted)

    def full(self, x: t.Tensor):
        contracted = self.contract(x)

        return {
            'contracted': contracted,
            'value': self.value(contracted),
            'grad': self.grad(contracted)
        }

    def value(self, contracted: List[t.Tensor]) -> t.Tensor:
        res = contracted[0]
        for c in contracted[1:]:
            res = t.bmm(res, c)
        return t.squeeze(t.squeeze(res, 1), 1)

    def grad(self, contracted: List[t.tensor]) -> t.Tensor:
        left_cumprods = cumprod(contracted)
        right_cumprods = rev_cumprod(contracted)

        return t.stack(
            [
                einsum('bij,jmk,bkl->bm', left_cumprods[i], G, right_cumprods[i + 1])
                for i, G in enumerate(self.Gs)
            ],
            1
        )

    def hess(self, contracted: Sequence[t.Tensor]) -> t.Tensor:
        products = [[None for _ in range(self.d + 1)] for _ in range(self.d + 1)]
        bs = contracted[0].shape[0]

        for i in range(self.d):
            products[i][i] = t.eye(contracted[i].shape[1]).repeat(bs, 1, 1)
            for j in range(i, self.d):
                products[i][j + 1] = t.bmm(products[i][j], contracted[j])
        products[self.d][self.d] = t.eye(1).repeat(bs, 1, 1)

        hess = [[None for _ in range(self.d)] for _ in range(self.d)]
        for i, G1 in enumerate(self.Gs):
            for j, G2 in enumerate(self.Gs):
                if i == j:
                    hess[i][j] = t.zeros(bs, self.m, self.m)
                    continue
                if j < i:
                    continue

                hess[i][j] = einsum(
                    'bij,jxk,bkl,lyn,bno->bxy',
                    products[0][i], G1, products[i + 1][j], G2, products[j + 1][self.d]
                )
                hess[j][i] = t.transpose(hess[i][j], 1, 2)

        return t.stack([t.stack(row, 1) for row in hess], 1)

    @property
    def device(self):
        return self.Gs[0].device

    def d_alpha_reg(self, alpha: int) -> t.Tensor:
        d = t.eye(self.m, device=self.device)
        for _ in range(alpha):
            d = d[:-1, :] - d[1:, :]
        return self.q_reg(t.matmul(d.T, d))

    def tv_reg(self) -> t.Tensor:
        return self.d_alpha_reg(1)

    def q_reg(self, D) -> t.Tensor:
        def cumprod(Gs):
            cp = [t.ones(1, 1, 1, 1, device=Gs[0].device)]
            for G in Gs:
                cp.append(einsum('ijkl,jax,kby,ab->ixyl', cp[-1], G, G, D))
            return cp

        def reverse(Gs):
            return [t.transpose(G, 0, -1) for G in Gs][::-1]

        left = cumprod(self.Gs)
        right = reverse(cumprod(reverse(self.Gs)))

        values = []
        for i, G in enumerate(self.Gs):
            values.append(einsum('ijkl,jxb,xy,kyc,abcd->', left[i], G, D, G, right[i + 1]))

        return t.sum(t.stack(values))

    def tensor(self):
        tensor = self.Gs[0]
        for G in self.Gs[1:]:
            tensor = t.tensordot(tensor, G, ((-1,), (0,)))
        return tensor[0, ..., 0]

    @property
    def full_tensor(self):
        result = self.Gs[0]
        for G in self.Gs[1:]:
            result = t.tensordot(result, G, ((-1,), (0,)))
        return result.squeeze(0).squeeze(-1)

    def two_to_matrices(self, main: Sequence[t.Tensor], new: Sequence[t.Tensor]) -> Sequence[t.Tensor]:
        """
            A := new
            b := main

                       |b1   0|   |b2    0|   |b3|
            [A0, b0]   |      |   |       |   |  |
                       |A1  b1|   |A2   b2|   |A3|

            [A0, b0] --> [A0 b1 + b0 A1, b0 b1] --> [A0 b1 b2 + b0 A1 b2 + b0 b1 A2, b0 b1 b2] --> ...
        """
        main = [m[..., None, None] for m in main]
        new = [n[..., None, None] for n in new]

        if self.n_dims == 1:
            bs = new
        else:
            b_first = t.cat([new[0], main[0]], -1)
            b_inner = [
                t.cat([t.cat([m, t.zeros_like(m)], -1), t.cat([n, m], -1)], -2)
                for m, n in zip(main[1:-1], new[1:-1])
            ]
            b_last = t.cat([main[-1], new[-1]], -2)
            bs = [b_first] + b_inner + [b_last]

        return bs

    def linear_one_contraction(self, bs: Sequence[t.Tensor], return_history: bool = False):
        return self.linear_matrix_contraction([b[..., None, None] for b in bs], return_history)

    def linear_two_contraction(
        self, main: Sequence[t.Tensor], new: Sequence[t.Tensor], return_history: bool = False
    ) -> t.Tensor:
        return self.linear_matrix_contraction(self.two_to_matrices(main, new), return_history)

    def linear_matrix_contraction(self, bs: Sequence[t.Tensor], return_history: bool = False):
        """
        bs: [n_dims][batchs..., m, rows, cols]

        x(2)---+---y(3)   matrix dimensions
               |
        i(0)--res--j(1)   G dimensions
        """

        assert len(bs) == self.n_dims

        res = t.ones(*bs[0].shape[:-3], 1, 1, 1, 1, device=self.Gs[0].device, dtype=self.Gs[0].dtype)

        if return_history:
            history = [res]

        for b, G in zip(bs, self.Gs):
            res = einsum('...ijxy,jmk,...myz->...ikxz', res, G, b)  # TODO: torch direct einsum is 2x faster WHY?
            if return_history:
                history.append(res)

        res = squeeze(res, (-4, -3, -2, -1))

        if return_history:
            return res, [squeeze(h, -4) for h in history]
        else:
            return res

    def square_one_contraction(self, Ds: Sequence[t.Tensor], return_history: bool = False) -> t.Tensor:
        return self.square_matrix_contraction([D[..., None, None] for D in Ds], return_history)

    def square_two_contraction(
            self, D_mains: Sequence[t.Tensor], D_news: Sequence[t.Tensor], return_history: bool = False
    ) -> t.Tensor:
        return self.square_matrix_contraction(self.two_to_matrices(D_mains, D_news), return_history)

    def square_matrix_contraction(self, Ds: Sequence[t.Tensor], return_history: bool = False):
        """
        Ds: [n_dims][batchs..., m, m, rows, cols]

        j(1)---+---l(3)   second G dimensions
               |
        i(0)--res--k(2)   first G dimensions
               |
          a(4)-+-b(5)     matrix dimensions
        """

        res = t.ones(*Ds[0].shape[:-4], 1, 1, 1, 1, 1, 1, device=self.Gs[0].device, dtype=self.Gs[0].dtype)

        if return_history:
            history = [res]

        for D, G in zip(Ds, self.Gs):
            res = einsum('...ijklab,kxm,lyn,...xybc->...ijmnac', res, G, G, D)
            if return_history:
                history.append(res)

        res = squeeze(res, (-6, -5, -4, -3, -2, -1))

        if return_history:
            return res, [squeeze(h, (-5, -6)) for h in history]
        else:
            return res


def tt_dot_product(tt1: TTNetwork, tt2: TTNetwork) -> t.Tensor:
    res = t.ones(1, 1, device=tt1.Gs[0].device)
    for G1, G2 in zip(tt1.Gs, tt2.Gs):
        res = einsum('ij,ikl,jkn->ln', res, G1, G2)
    return squeeze(res, (0, 1))


def cumprod(matrices: Sequence[t.Tensor]) -> List[t.Tensor]:
    res = [t.eye(matrices[0].shape[1], device=matrices[0].device).repeat(matrices[0].shape[0], 1, 1)]
    for m in matrices:
        res.append(t.bmm(res[-1], m))
    return res


def rev_cumprod(matrices: Sequence[t.Tensor]) -> List[t.Tensor]:
    def reverse_transpose(ms):
        return [m.transpose(1, 2) for m in ms][::-1]
    return reverse_transpose(cumprod(reverse_transpose(matrices)))


def pad_to_rang(tt: TTNetwork, rang: int):
    Gs = []
    for i, G in enumerate(tt.Gs):
        l_pad = 0 if i == 0 else max(0, rang - G.shape[0])
        r_pad = 0 if i == len(tt.Gs) - 1 else max(0, rang - G.shape[-1])
        Gs.append(nn.functional.pad(G, [0, r_pad, 0, 0, 0, l_pad], 'constant', 0.))

    return TTNetwork(Gs, to_param_list=True)

