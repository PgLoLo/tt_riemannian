import pytest

import torch as t

from dataset_reduction.score.tt.core.tt_network import TTNetwork
from dataset_reduction.score.tt.core.rounding import round_tt
from dataset_reduction.score.tt.test_common import assert_tt_equal
from dataset_reduction.score.tt.utils import lists_to_rang2_concat

N_DIMS = [3]
M = [8]
R = [2]


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_rounding_with_the_same_rang(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)
    tt_rounded = round_tt(tt, r)
    assert_tt_equal(tt, tt_rounded)


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_rounding_with_the_same_rang(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)
    tt_rounded = round_tt(tt, r)
    assert_tt_equal(tt, tt_rounded)


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
def test_rounding_for_laplace_like_tensor(n_dims: int, m: int):
    print()
    base = [t.randn(1, m, 1) for _ in range(n_dims)]
    moving = [t.randn(1, m, 1) for _ in range(n_dims)]
    zeros = t.zeros(1, m, 1)

    high_rang_Gs = []
    for i, (b, mv) in enumerate(zip(base, moving)):
        if i == 0:
            high_rang_Gs.append(t.cat([mv] + [b] * (n_dims - 1), 2))
        elif i == n_dims - 1:
            high_rang_Gs.append(t.cat([b] * (n_dims - 1) + [mv], 0))
        else:
            rows = []
            for j in range(n_dims):
                curr_value = mv if i == j else b
                rows.append(t.cat([zeros] * j + [curr_value] + [zeros] * (n_dims - j - 1), 2))
            high_rang_Gs.append(t.cat(rows, 0))
    high_rang_tt = TTNetwork(high_rang_Gs)

    low_rang_Gs = lists_to_rang2_concat(moving, base, base, 0, 2)
    low_rang_tt = TTNetwork(low_rang_Gs)
    rounded_tt = round_tt(high_rang_tt, 2)

    assert_tt_equal(low_rang_tt, high_rang_tt)
    assert_tt_equal(rounded_tt, high_rang_tt)
    assert_tt_equal(rounded_tt, low_rang_tt)



