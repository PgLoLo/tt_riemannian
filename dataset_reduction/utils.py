from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt


def parabola_min(grad, hess):
    return -grad / (2 * hess)


def plot_mean_plots(data: Dict[Any, np.ndarray], factor: float = 2.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    for key, data in data.items():
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) / np.sqrt(len(data))
        xs = np.arange(len(mean))

        ax1.plot(xs, mean, label=str(key))
        ax1.fill_between(xs, mean - factor * std, mean + factor * std, alpha=0.2)
        ax1.legend()

        n = len(xs) // 2
        xs = xs[n:]
        mean = mean[n:]
        std = std[n:]

        ax2.plot(xs, mean)
        ax2.fill_between(xs, mean - factor * std, mean + factor * std, alpha=0.2)

    plt.show()


def suffix_with_date(folder: Path) -> Path:
    now = str(datetime.now()).replace(' ', '_')
    return folder / now