import torch as t

from dataset_reduction.score.tt.core.tt_network import tt_dot_product, TTNetwork


def assert_tt_equal(tt1: TTNetwork, tt2: TTNetwork):
    assert t.allclose(tt_dot_product(tt1, tt2)**2, tt_dot_product(tt1, tt1) * tt_dot_product(tt2, tt2))
