import torch as t

from dataset_reduction.dl_routine import list_to_device
from dataset_reduction.score.tt.core.tt_network import TTNetwork, tt_dot_product
from dataset_reduction.score.tt.core.riemannian import prepare_for_AD_in_tangent, apply_gauge_on_grad
from dataset_reduction.score.tt.core.orthogonalization import deltas_to_tangent
from dataset_reduction.score.tt.core.rounding import round_tt


def step(lr: float, rang: int, loss_fn, tt_net: TTNetwork):
    rs, tt_tangent, left, right, s = prepare_for_AD_in_tangent(tt_net)

    loss = loss_fn(tt_tangent)
    loss.backward()

    with t.no_grad():
        grads = apply_gauge_on_grad(left, rs)

    with t.no_grad():
        left = list_to_device(left, t.device('cpu'))
        right = list_to_device(right, t.device('cpu'))
        rs_new = list_to_device([r - lr * grad for r, grad in zip(rs, grads)], t.device('cpu'))
        tangent_with_grad = deltas_to_tangent(left, right, rs_new)

        rounded = round_tt(tangent_with_grad, rang)

    rounded = TTNetwork(rounded.Gs, to_param_list=True)

    return rounded, loss.item()


def loss_fn(tt_net: TTNetwork) -> t.Tensor:
    return (tt_dot_product(tt_net, tt_net) - 1)**2


tt_network = TTNetwork.randomly_generated(2, 1, 3)

while True:
    tt_network, loss = step(1e-3, 1, loss_fn, tt_network)
    print(loss)





