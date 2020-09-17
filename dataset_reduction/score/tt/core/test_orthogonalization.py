from typing import Sequence

import pytest

import torch as t

from dataset_reduction.score.tt.core.tt_network import TTNetwork
from dataset_reduction.score.tt.core.orthogonalization import orthogonalize, prepare_for_tangent, deltas_to_tangent, \
    get_orthogonalized
from dataset_reduction.score.tt.test_common import assert_tt_equal
from dataset_reduction.score.tt.utils import reverse_and_transpose


N_DIMS = [4]
M = [8]
R = [4]


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_prepare_for_tangent(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)

    left, right, s = prepare_for_tangent(tt)
    for i in range(n_dims):
        tt_new = TTNetwork(left[:i] + [s[i]] + right[i + 1:])
        assert_tt_equal(tt, tt_new)


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_deltas_to_tangent_space(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)

    left, right, s = prepare_for_tangent(tt)
    for i in range(n_dims):
        s_new = [t.zeros_like(v) for v in s]
        s_new[i] = s[i]
        tt_new = deltas_to_tangent(left, right, s_new)
        assert_tt_equal(tt, tt_new)


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_left_right_interchange_in_orthogonalize(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)

    q_left, r_left = orthogonalize(tt)
    q_right, r_right = orthogonalize(tt.construct_reverse(), right_to_left=True)
    q_right = reverse_and_transpose(q_right)
    r_right = reverse_and_transpose(r_right)

    def compare(seq1: Sequence[t.Tensor], seq2: Sequence[t.Tensor]):
        assert len(seq1) == len(seq2)
        for s1, s2 in zip(seq1, seq2):
            assert t.allclose(s1, s2)

    compare(q_right, q_left)
    compare(r_right, r_left)


@pytest.mark.parametrize('n_dims', N_DIMS)
@pytest.mark.parametrize('m', M)
@pytest.mark.parametrize('r', R)
def test_in_get_orthogonalization(n_dims: int, m: int, r: int):
    tt = TTNetwork.randomly_generated(n_dims, r, m)

    left_to_right = get_orthogonalized(tt)
    right_to_left = get_orthogonalized(tt, right_to_left=True)

    assert_tt_equal(tt, left_to_right)
    assert_tt_equal(left_to_right, right_to_left)
