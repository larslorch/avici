import os
import matplotlib.pyplot as plt
import imageio


def visualize_ground_truth(mat, size=4.0):
    """    
    `mat`: (d, d) 
    """
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, vmin=0, vmax=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(r'Ground truth $G^*$', pad=10)
    plt.show()
    return


def visualize(mats, t, save_path=None, n_cols=7, size=2.5, show=False):
    """
    Based on visualization by https://github.com/JannerM/gamma-models/blob/main/gamma/visualization/pendulum.py
    
    `mats` should have shape (N, d, d) and take values in [0,1]
    """

    N = mats.shape[0]
    n_rows = N // n_cols
    if N % n_cols:
        n_rows += 1
    
    plt.rcParams['figure.figsize'] = [size * n_cols, size * n_rows]
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.flatten()

    # for j, (ax, mat) in enumerate(zip(axes[:len(mats)], mats)):
    for j, ax in enumerate(axes): 
        if j < len(mats):
            # plot matrix of edge probabilities
            ax.matshow(mats[j], vmin=0, vmax=1)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_title(r'$Z^{('f'{j}'r')}$', pad=3)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.axis('off')
    

    # save
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + f'/img{t}.png')
        img = imageio.imread(save_path + f'/img{t}.png')
    else:
        img = None
    if show:
        plt.show()
    plt.close()
    return img
        
