import matplotlib.pyplot as plt
import numpy as np

def visualize(mat, true=None, size=0.75):
    size = np.sqrt(mat.shape[-1]) * size
    plt.rcParams['figure.figsize'] = [size if true is None else 2 * size, size]
    if true is None:
        fig, axs = plt.subplots(1, 1)
        mats, labels = [mat],  ["predicted"]
    else:
        fig, axs = plt.subplots(1, 2)
        mats, labels = [mat, true],  ["predicted", "true"]

    for ax, label, mat in zip(axs, labels, mats):
        ax.matshow(mat, vmin=0, vmax=1)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_title(label, pad=10)

    plt.tight_layout()
    plt.show()
    return

