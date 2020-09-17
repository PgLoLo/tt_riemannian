from functools import lru_cache
from typing import Sequence, List

import torch as t
from opt_einsum import contract_expression


def add_matrix_dimensions(vs: Sequence[t.Tensor]):
    return [v[..., None, None] for v in vs]


def lists_to_rang2_matrix(
    main: Sequence[t.Tensor], upper: Sequence[t.Tensor], lower: Sequence[t.Tensor]
) -> List[t.Tensor]:
    """
    Converts input tensors into matrices with tensors as values int the following way:
    U := upper
    L := lower
    S := main

               |U1   0|   |U2    0|   |U3|
    [S0, L0]   |      |   |       |   |  |
               |S1  L1|   |S2   L2|   |S3|

    """

    main = add_matrix_dimensions(main)
    upper = add_matrix_dimensions(upper)
    lower = add_matrix_dimensions(lower)

    if len(main) == 1:
        return main
    else:
        first = t.cat([main[0], lower[0]], -1)
        inner = [
            t.cat([t.cat([u, t.zeros_like(u)], -1), t.cat([m, l], -1)], -2)
            for u, l, m in zip(upper[1:-1], lower[1:-1], main[1:-1])
        ]
        b_last = t.cat([upper[-1], main[-1]], -2)
        return [first] + inner + [b_last]


def lists_to_rang2_concat(
    main: Sequence[t.Tensor], upper: Sequence[t.Tensor], lower: Sequence[t.Tensor], row_axis: int, col_axis: int
) -> Sequence[t.Tensor]:
    """
    Concatenates input tensors in the following way:

    U := upper
    L := lower
    S := main

               |U1   0|   |U2    0|   |U3|
    [S0, L0]   |      |   |       |   |  |
               |S1  L1|   |S2   L2|   |S3|
    """

    if len(main) == 1:
        return [main[0]]
    else:
        first = t.cat([main[0], lower[0]], col_axis)
        inner = [
            t.cat([t.cat([u, t.zeros_like(u)], col_axis), t.cat([m, l], col_axis)], row_axis)
            for u, l, m in zip(upper[1:-1], lower[1:-1], main[1:-1])
        ]
        b_last = t.cat([upper[-1], main[-1]], row_axis)
        return [first] + inner + [b_last]


def reverse_and_transpose(tensors: Sequence[t.Tensor]) -> List[t.Tensor]:
    return [tensor.permute(list(range(tensor.dim())[::-1])) for tensor in tensors[::-1]]


@lru_cache
def cached_einsum_expr(expr: str, *shapes: t.Size):
    return contract_expression(expr, *shapes)


def einsum(expr: str, *args: t.Tensor):
    compiled = cached_einsum_expr(expr, *[arg.shape for arg in args])
    return compiled(*args)