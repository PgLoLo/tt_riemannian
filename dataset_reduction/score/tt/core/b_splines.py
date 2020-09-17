from functools import lru_cache
from typing import List

import torch as t
import numpy as np
from numpy.polynomial import Polynomial
from scipy.linalg import toeplitz
from torch import nn


__all__ = ['BSpline', 'BSplinePhi']


def build_basis(h: float, q: int) -> List[Polynomial]:
    o = Polynomial([0.])
    e = Polynomial([1.])
    x = Polynomial([0., 1.])

    B = np.zeros((q + 1, q + 1), dtype=object)  # B[p, i, j] -- coeffitient of x^j in A_i for p-th spline
    for i in range(q + 1):
        for j in range(q + 1):
            B[i, j] = o

    B[0, 0] = e
    for p in range(1, q + 1):
        for i in range(p + 1):
            B[p, i] += B[p - 1, i] * x
            if i:
                B[p, i] += B[p - 1, i - 1](x - h) * (h * (p + 1) - x)
            B[p, i] /= h * p

    return [B[-1, i] for i in range(q + 1)]


def deriv_integral(q: int, h: float, B: List[Polynomial], p: int, left: int = 0, right: int = None):
    if right is None:
        right = q

    res = 0
    for i in range(left, right + 1):
        l, r = i * h, i * h + h
        integ = B[i].deriv(p).integ()
        res += integ(r) - integ(l)

    return res


def build_deriv_integrals_l2(q: int, h: float, B: List[Polynomial], p: int, left: int = 0, right: int = None):
    if right is None:
        right = q

    x = Polynomial([0., 1.])

    c = []
    for i in range(-q, q + 1):
        curr = 0
        for j in range(max(left, i), min(right, i + q) + 1):
            l, r = j * h, j * h + h
            first = B[j - i](x - i * h)
            second = B[j]
            integ = (first.deriv(p) * second.deriv(p)).integ()

            curr += integ(r) - integ(l)
        c.append(curr)

    return c


class BSpline(nn.Module):
    def __init__(self, h: float, q: int, t0: float, n: int):
        super().__init__()

        self.h = h
        self.q = q
        self.t0 = t0
        self.n = n

        self.B_np = build_basis(h, q)

        self.buffer_from_polynomes('B', self.B_np)
        self.buffer_from_polynomes('B_grad', [b.deriv() for b in self.B_np])
        self.buffer_from_polynomes('B_hess', [b.deriv(2) for b in self.B_np])
        self.buffer_from_polynomes('B_int', [b.integ(1, 0, i * self.h) for i, b in enumerate(self.B_np)])

    def buffer_from_polynomes(self, name: str, polynomes):
        self.register_buffer(name, t.tensor([poly.coef for poly in polynomes[::-1]]).float())

    @lru_cache
    def l2_deriv_D(self, p: int, change_bounds: bool):
        c = build_deriv_integrals_l2(self.q, self.h, self.B_np, p)
        c, r = c[:self.q + 1][::-1], c[self.q:]
        c = np.concatenate([c, np.zeros(self.n - self.q - 1)])
        r = np.concatenate([r, np.zeros(self.n - self.q - 1)])

        D = toeplitz(c, r)

        if change_bounds:
            for i in range(self.q):
                c = build_deriv_integrals_l2(self.q, self.h, self.B_np, p, left=self.q - i)
                D[i, :self.q + i + 1] = c[-self.q - i - 1:]
            for i in range(self.q):
                c = build_deriv_integrals_l2(self.q, self.h, self.B_np, p, right=i)
                D[-1 - i, -self.q - i - 1:] = c[:self.q + i + 1]

        return t.tensor(D, device=self.device, dtype=self.dtype)

    @lru_cache
    def int_deriv_coeffs(self, p: int, change_bounds: bool):
        """
        int f^{(p)}(x) dx
        """
        c = t.ones(self.n, device=self.device, dtype=self.dtype) * deriv_integral(self.q, self.h, self.B_np, p)
        if change_bounds:
            for i in range(self.q):
                c[i] = deriv_integral(self.q, self.h, self.B_np, p, left=self.q - i)
            for i in range(self.q):
                c[-i - 1] = deriv_integral(self.q, self.h, self.B_np, p, right=i)
        return c

    def int_to_x_coeffs(self, xs: t.Tensor):
        """
        int_{-inf}^x f(x') dx'
        """
        assert not xs.requires_grad

        cum_ints = t.tensor(
            [deriv_integral(self.q, self.h, self.B_np, 0, right=self.q - i - 1) for i in range(self.q + 1)],
            device=xs.device,
        )

        const_value = deriv_integral(self.q, self.h, self.B_np, 0)
        xs = xs - self.t0
        first_i = t.floor(xs / self.h).to(t.int32)

        cols = first_i[:, None] + t.arange(self.q + 1, device=self.device)[None, :]
        rows = t.arange(len(xs), device=self.device)[:, None].repeat(1, self.q + 1)
        inds = t.stack([rows, cols], 0).reshape(2, -1)

        xs = xs - first_i * self.h
        xs = xs[:, None] + self.h * (self.q - t.arange(self.q + 1, device=self.device))[None, :]
        xs_p = t.stack([xs ** i for i in range(self.q + 2)], 2)
        values = self.eval_on_coeffs(xs_p, self.B_int)
        values += cum_ints[None, :]

        mask, = t.where((0 <= inds[1, :]) & (inds[1, :] < self.n))

        result = self.values_inds_to_tensor(len(xs), inds, values, mask)
        result = t.where(t.arange(self.n, device=self.device)[None, :].repeat(len(xs), 1) < first_i[:, None], t.ones_like(result) * const_value, result)

        return result

    @property
    def device(self):
        return self.B.device

    @property
    def dtype(self):
        return self.B.dtype

    @staticmethod
    def eval_on_coeffs(xs_poly, B):
        return (xs_poly[:, :, :B.shape[1]] * B[None, :, :]).sum(2)

    def eval_spline(self, xs, value_only: bool = True):
        xs = xs[:, None] + self.h * (self.q - t.arange(self.q + 1, device=self.device))[None, :]
        xs_p = t.stack([xs ** i for i in range(self.q + 1)], 2)
        if value_only:
            return self.eval_on_coeffs(xs_p, self.B)
        else:
            return {
                'value': self.eval_on_coeffs(xs_p, self.B),
                'grad': self.eval_on_coeffs(xs_p, self.B_grad),
                'hess': self.eval_on_coeffs(xs_p, self.B_hess)
            }

    def values_inds_to_tensor(self, n_xs, inds, values, mask):
        inds = inds[:, mask]
        values = values.reshape(-1)[mask]
        return t.sparse_coo_tensor(inds, values, (n_xs, self.n), device=self.device).to_dense()

    def forward(self, xs, value_only: bool = True):
        xs = xs - self.t0
        first_i = t.floor(xs / self.h).to(t.int32)

        cols = first_i[:, None] + t.arange(self.q + 1, device=self.device)[None, :]
        rows = t.arange(len(xs), device=self.device)[:, None].repeat(1, self.q + 1)
        inds = t.stack([rows, cols], 0).reshape(2, -1)

        xs_minus_first_i = xs - first_i * self.h
        values = self.eval_spline(xs_minus_first_i, value_only)
        mask, = t.where((0 <= inds[1, :]) & (inds[1, :] < self.n))

        if value_only:
            return self.values_inds_to_tensor(len(xs), inds, values, mask)
        else:
            return {key: self.values_inds_to_tensor(len(xs), inds, value, mask) for key, value in values.items()}


class BSplinePhi(nn.Module):
    def __init__(self, q: int, l: float, r: float, n: int):
        super().__init__()
        self.l = l
        self.r = r
        self.b_spline = BSpline((r - l) / (n - q), q, l, n)

    @property
    def n(self):
        return self.b_spline.n

    @property
    def device(self):
        return self.b_spline.device

    @property
    def dtype(self):
        return self.b_spline.dtype

    def reshape(self, value, shape):
        return value.reshape(*shape, self.b_spline.n)

    def forward(self, xs, value_only: bool = True):
        shape = xs.shape
        xs = xs.reshape(-1)
        result = self.b_spline(xs, value_only)

        if value_only:
            return self.reshape(result, shape)
        else:
            return {key: self.reshape(value, shape) for key, value in result.items()}


