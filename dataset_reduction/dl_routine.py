from contextlib import contextmanager
from pathlib import Path
from typing import List, Union, Sequence, Tuple

import numpy as np
import torch as t
import torchvision
from torch import nn
# from torch import einsum
from opt_einsum import contract
import torch.utils.data
import torch.utils.tensorboard
from tqdm.autonotebook import trange
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import ipyvolume as ipv

from dataset_reduction.utils import suffix_with_date


__all__ = ['to_tensor']


def to_tensor(data):
    if t.is_tensor(data):
        return data
    else:
        return t.tensor(data)


class TensorDataset:
    class TestLoader:
        def __init__(self, dataset, batch_sz):
            self.batch_sz = batch_sz
            self.dataset = dataset

        def __iter__(self):
            for offset in range(0, len(self.dataset), self.batch_sz):
                yield self.dataset.X[offset : offset + self.batch_sz], self.dataset.y[offset : offset + self.batch_sz]

    def __init__(self, X, y, device=t.device('cpu')):
        self.X = to_tensor(X)
        self.y = to_tensor(y)
        self.device = None
        self.to(device)

    def to(self, device: t.device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        self.device = device
        return self

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item: int):
        return self.X[item], self.y[item]

    def train_iterator(self, batch_sz: int):
        while True:
            inds = t.randint(0, len(self), (batch_sz,), device=self.X.device)
            yield self.X[inds], self.y[inds]

    def test_loader(self, batch_sz: int):
        return self.TestLoader(self, batch_sz)

    def filter_classes(self, classes: Sequence[int]):
        new_X = []
        new_y = []

        for i, cls in enumerate(classes):
            inds = self.y == cls
            new_X.append(self.X[inds])
            new_y.append(t.ones(sum(inds), dtype=t.int64) * i)

        X = t.cat(new_X, axis=0)
        y = t.cat(new_y, axis=0)

        return TensorDataset(X, y)

    def combine_classes(self, classes: Sequence[Sequence[int]]):
        new_X = []
        new_y = []

        for i, cls in enumerate(classes):
            for j in cls:
                inds = self.y == j
                new_X.append(self.X[inds])
                new_y.append(t.ones(sum(inds), dtype=t.int64) * i)

        X = t.cat(new_X, axis=0)
        y = t.cat(new_y, axis=0)

        return TensorDataset(X, y)

    def apply_pca(self, pca: PCA) -> 'TensorDataset':
        return TensorDataset(pca.transform(full_detach(self.X).reshape(len(self), -1)), self.y, self.device)


class DimensionalityReduction:
    def __init__(self, dataset: TensorDataset, n_dims: int):
        self.shape = dataset.X.shape[1:]

        reshaped = dataset.X.reshape(dataset.X.shape[0], -1)
        self.mean = reshaped.mean(0)
        shifted = reshaped - self.mean[None, :]
        U, S, V = np.linalg.svd(full_detach(shifted))

        new_X = U[:, :n_dims]
        transform_matrix = S[:n_dims, None] * V[:n_dims, :]
        self.transform_matrix = t.tensor(transform_matrix, device=dataset.X.device, dtype=dataset.X.dtype)
        self.dataset = TensorDataset(t.tensor(new_X, device=dataset.X.device, dtype=dataset.X.dtype), dataset.y)

    def transform(self, X):
        return (t.matmul(X, self.transform_matrix) + self.mean[None, :]).reshape(-1, *self.shape)


def pca(train: TensorDataset, test: TensorDataset, n_dims: int) -> Tuple[TensorDataset, TensorDataset]:
    X = full_detach(train.X).reshape(len(train.X), -1)
    pca = PCA(n_dims).fit(X)

    return train.apply_pca(pca), test.apply_pca(pca)


def cifar(root: Union[str, Path], train: bool = True) -> TensorDataset:
    dataset = torchvision.datasets.CIFAR10(root, train, download=True)
    return TensorDataset(np.transpose(dataset.data, (0, 3, 1, 2)).astype(np.float32) / 255, dataset.targets)


def mnist_dataset(root: Union[str, Path], train: bool = True) -> TensorDataset:
    mnist = torchvision.datasets.MNIST(root, train, download=True)
    X = mnist.data.reshape((-1, 1, 28, 28)).float() / 255
    y = mnist.targets

    X.to(t.device('cpu'))
    y.to(t.device('cpu'))

    return TensorDataset(X, y)


def fashion_mnist_dataset(root: Union[str, Path], train: bool = True) -> TensorDataset:
    mnist = torchvision.datasets.FashionMNIST(root, train, download=True)
    X = mnist.data.reshape((-1, 1, 28, 28)).float() / 255
    y = mnist.targets

    return TensorDataset(X, y)


def repeat_iterator(iterator):
    while True:
        yield from iterator


def fc(dims: List[int]):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers[:-1])


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def mnist_network_supersmall(out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 2, 3, 2, 1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(392, out_dim)
    )


def mnist_network_small(out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(8, 16, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, stride=2),
        Flatten(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, out_dim),
    )


class SoftmaxToBinaryWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: t.Tensor):
        logits = self.model(x)
        assert logits.shape[-1] == 2

        logits = logits[..., 1]
        return t.stack([t.zeros_like(logits), logits], axis=-1)


def mnist_network_large() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def softmax_regression_model(n_in: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(
        Flatten(),
        nn.Linear(n_in, n_out),
    )


class LogisticRegressionModel(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()

        self.n_in = n_in
        self.theta = nn.Parameter(t.randn(n_in + 1), True)
        self.flatten = Flatten()

    @classmethod
    def from_w_b(cls, w, b):
        w = to_tensor(w).float()
        b = to_tensor([b]).float()

        model = cls(len(w))
        model.theta.data.copy_(t.cat([w, b]))

        return model

    @property
    def w(self):
        return self.theta[:-1]

    @property
    def b(self):
        return self.theta[-1]

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.flatten(x)
        logits = t.matmul(x, self.w[:, None]) + self.b
        return t.cat([t.zeros_like(logits), logits], axis=-1)


def to_inf_data_loader(dataset, batch_sz):
    dataloader = t.utils.data.DataLoader(dataset, batch_sz, shuffle=True, num_workers=0)
    return repeat_iterator(dataloader)


class RegularizedSMCE(nn.Module):
    def __init__(self, prob : float = 0.95):
        super().__init__()
        self.prob = prob

    def forward(self, logits: t.Tensor, y: t.Tensor) -> t.Tensor:
        log_p = t.log_softmax(logits, axis=-1)

        target = t.ones_like(log_p) * (1 - self.prob) / (log_p.shape[1] - 1)
        target[t.arange(len(target)), y] = self.prob

        loss = -(target * log_p).sum()
        return loss


class Trainer:
    def __init__(
        self,
        loss_fn,
        lr: float,
        train_batch_sz: int,
        test_batch_sz: int,
        device: t.device = t.device('cuda:0'),
    ):
        self.loss_fn = loss_fn
        self.lr = lr
        self.batch_sz = train_batch_sz
        self.test_batch_sz = test_batch_sz
        self.device = device

    def train(self, train_data, test_data, model, n_itrs, log_dir: Path, *, opt_kwargs={'weight_decay': 0e-3}):
        model.to(self.device)

        with t.utils.tensorboard.SummaryWriter(suffix_with_date(log_dir)) as writer:
            train_itr = train_data.train_iterator(self.batch_sz)
            test_data_loader = test_data.test_loader(self.test_batch_sz)

            optimizer = t.optim.Adam(model.parameters(), self.lr, **opt_kwargs)
            test_accs = []

            for step in trange(n_itrs, desc='training', leave=False):
                loss = self.forward(model, train_itr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('train/loss', loss, step)

                writer.add_scalar('other/grad_sqr_norm', sum(param.grad.norm() for param in model.parameters()), step)

                if step % 250 == 0:
                    test_acc = self.full_evaluation(model, test_data_loader)
                    writer.add_scalar('test/acc', test_acc, step)
                    test_accs.append(test_acc.item())

                    writer.add_scalar('other/model_sqr_norm', sum(param.norm()**2 for param in model.parameters()), step)

            return test_accs

    def forward(self, model, loader):
        X, y = next(loader)
        logits = model(X)
        return self.loss_fn(logits, y)

    def full_evaluation(self, model, data_loader):
        model.eval()

        acc = 0

        for X, y in data_loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = model(X)
            preds = t.argmax(logits, axis=-1)

            acc += (preds == y).sum()

        model.train()

        return acc.float() / len(data_loader.dataset)


def color_2d(model, x1s, x2s, device=t.device('cpu')):
    X1, X2 = np.meshgrid(x1s, x2s)
    X = np.stack([X1.flatten(), X2.flatten()], axis=-1)
    X = t.tensor(X, dtype=t.float32).to(device)
    ys = model.to(device)(X)
    ys = ys.detach().cpu().numpy().reshape(X1.shape)
    return X1, X2, ys


def full_detach(tensor: t.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class DistrDataset(nn.Module):
    def __init__(self, distr: t.distributions.Distribution, device: t.device = t.device('cpu')):
        super().__init__()
        self.distr = distr
        self.to(device)

    def cuda(self):
        self.device = t.device('cuda:0')
        return self

    def to(self, device: t.device):
        self.device = device
        return self

    def bath_iterator(self, batch_sz: int):
        while True:
            yield self.distr.sample([batch_sz]).to(self.device)

    def log_prob(self, x):
        return self.distr.log_prob(x).to(self.device)

    train_iterator = bath_iterator


class NotNormal(nn.Module):
    def __init__(self, noise: float, d: float):
        self.noise = noise
        self.d = d

    def sample(self, shape):
        xs = t.randn(*shape, 1) * self.noise**2
        xs = t.where(xs < 0, -self.d + xs**2, +self.d - xs**2)
        return xs


def to_tensor(array: np.ndarray, dtype=t.float, device=t.device('cpu')):
    return t.tensor(array, dtype=dtype, device=device)


class Exp(nn.Module):
    def forward(self, x):
        return x.exp()


class Log(nn.Module):
    def forward(self, x):
        return t.where(x <= 0, 1e-9 * t.ones_like(x), x).log()


def squeeze(tensor, dims):
    if isinstance(dims, int):
        return tensor.squeeze(dims)

    dims = [dim if dim >= 0 else tensor.dim() + dim for dim in dims]
    for d in sorted(dims)[::-1]:
        tensor = tensor.squeeze(d)
    return tensor


def device(model: nn.Module):
    # return model.tt.Gs[0].device  # TODO: LOL
    return next(model.parameters()).device


def scatter_nd(xs: t.Tensor):
    xs = full_detach(xs)

    if xs.shape[1] == 1:
        plt.hist(xs[:, 0], bins=100)
    elif xs.shape[1] == 2:
        plt.scatter(*xs.T, s=.1)
    elif xs.shape[1] == 3:
        ipv.clear()
        ipv.scatter(*xs.T)
        ipv.show()
    else:
        assert False, f"Wrong dimensionality: {xs.shape=}"


def sqr_norm(g, dims):
    return (g ** 2).sum(dims)


@contextmanager
def on_device(module: nn.Module, device: t.device):
    device_before = next(module.parameters()).device
    module.to(device)

    try:
        yield None
    finally:
        module.to(device_before)


@contextmanager
def on_cpu(module: nn.Module):
    device_before = next(module.parameters()).device
    module.cpu()

    try:
        yield None
    finally:
        module.to(device_before)
    # return on_device(module, t.device('cpu'))


@contextmanager
def on_cuda(module: nn.Module):
    return on_device(module, t.device('cuda'))


def list_to_device(tensors: Sequence[t.Tensor], device: t.device) -> List[t.Tensor]:
    return [t.to(device) for t in tensors]