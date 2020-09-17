import pytest

import torch as t

from dataset_reduction.score.tt.core.tt_network import TTNetwork, tt_dot_product

N_DIMS = [1, 2, 3]
M = [8]
R = [1, 4, 8, 16]


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_tt_dot_product_with_full(n_dims: int, m: int, r: int):
    tt1 = TTNetwork.randomly_generated(n_dims, r, m)
    tt2 = TTNetwork.randomly_generated(n_dims, r, m)

    assert t.allclose(tt_dot_product(tt1, tt2), (tt1.full_tensor * tt2.full_tensor).sum())
