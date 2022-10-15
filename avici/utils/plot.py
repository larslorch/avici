import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import imageio
import wandb
import numpy as np

import pandas as pd
from pandas.plotting._matplotlib.tools import _subplots

from avici.definitions import PROJECT_DIR


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


def visualize_mats(mats, colnames, rownames, max_cols=None, max_rows=None, size=1, show=False, diff=False):

    ncols = len(mats[0])
    if max_cols is not None:
        ncols = min(max_cols, ncols)

    nrows = len(mats)
    if max_rows is not None:
        nrows = min(max_rows, nrows)

    plt.rcParams['figure.figsize'] = [size * ncols, size * nrows]
    fig, axes = plt.subplots(nrows, ncols)

    if diff:
        # make predicted matrices predict difference to mean
        npreds = ncols - 2
        if npreds > 1:
            for i in range(nrows):
                # calculate mean prediction
                mu = np.zeros_like(mats[i][0])
                for j in range(2, npreds + 2):
                    mu += mats[i][j] / npreds

                # subtract mean from all matrices
                for j in range(2, npreds + 2):
                    mats[i][j] -= mu

    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            if i == 0:
                ax.set_title(colnames[j], pad=3)
            if j == 0:
                ax.set_ylabel(rownames[i])


            if diff and j != 0:
                # ax.matshow(mats[i][j], vmin=-0.5, vmax=0.5, cmap="seismic")
                matim = ax.matshow(mats[i][j], vmin=-0.5, vmax=0.5, cmap="Spectral")
                # matim = ax.matshow(mats[i][j], vmin=-0.5, vmax=0.5, cmap="hsv")

            else:
                matim = ax.matshow(mats[i][j], vmin=0, vmax=1, cmap="viridis")
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.axis('off')

    plt.tight_layout()

    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(matim, cax=cbar_ax)

    if show:
        plt.show()
    log_img = wandb.Image(plt)
    plt.close()
    return log_img


def visualize_pred(mats_prob, mats_pred, mats_true, max_cols=None, size=2.5, show=False):
    """
    Visualizes various quantities of prediction
    """
    if max_cols is not None:
        mats_prob, mats_pred, mats_true = mats_prob[:max_cols], mats_pred[:max_cols], mats_true[:max_cols]
    n_cols = mats_true.shape[0]

    plt.rcParams['figure.figsize'] = [size * n_cols, size * 3]
    fig, axes = plt.subplots(3, n_cols)
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            if i == 0:
                ax.matshow(mats_prob[j], vmin=0, vmax=1)
                ax.set_title(f'pred_probs_{j}', pad=3)
            elif i == 1:
                ax.matshow(mats_pred[j], vmin=0, vmax=1)
                ax.set_title(f'pred_{j}', pad=3)
            elif i == 2:
                ax.matshow(mats_true[j], vmin=0, vmax=1)
                ax.set_title(f'true_{j}', pad=3)
            # elif i in [3, 4]:
            #
            #     axis_, agg_style_ = (0, "in") if i == 3 else (1, "out")
            #
            #     for degrees, label, c, style_ in [(mats_pred[j].sum(axis_), "pred", "blue", 's-'),
            #                                       (mats_true[j].sum(axis_), "true", "black", 's-')]:
            #         degs, counts = np.unique(degrees, return_counts=True)
            #         freqs = counts / counts.sum()
            #         srt = np.argsort(degs)
            #
            #         ax.loglog(degs[srt], freqs[srt], style_, c=c, markersize=2, label=label)
            #
            #     ax.set_title(f'{agg_style_}-degree_{j}', pad=3)
            #     # ax.set_xlabel("degree")
            #     # ax.set_xlabel("frequency")


            if i in [0, 1, 2]:
                ax.tick_params(axis='both', which='both', length=0)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.axis('off')

    plt.tight_layout()
    if show:
        plt.show()
    log_img = wandb.Image(plt)
    plt.close()
    return log_img



def _scatter_matrix(
    frame,
    drop_cols=None,
    alpha=0.5,
    size_per_var=1.0,
    ax=None,
    diagonal="hist",
    marker=".",
    density_kwds=None,
    hist_kwds=None,
    range_padding=0.05,
    max_rows=None,
    max_cols=None,
    color_mask=None,
    cmain="black",
    **kwds,
):
    """Adapted from pandas version"""

    df = frame._get_numeric_data()
    if drop_cols is None:
        drop_cols = np.zeros(df.columns.size).astype(bool)

    n = np.logical_not(drop_cols).sum()
    n_rows = min(max_rows, n) if max_rows is not None else n
    n_cols = min(max_cols, n) if max_cols is not None else n
    naxes = n_rows * n_cols
    layout = (n_rows, n_cols)

    figsize = (n_cols * size_per_var, n_rows * size_per_var)

    # plt.rcParams['figure.figsize'] = figsize
    fig, axes = _subplots(naxes=naxes, figsize=figsize, ax=ax, layout=layout, squeeze=False)

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = pd.core.dtypes.missing.notna(df)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # GH 14855
    kwds.setdefault("edgecolors", "none")

    boundaries_list = []
    for idx in np.where(~drop_cols)[0]:
        values = df.iloc[:, idx].values[mask[df.columns[idx]].values]
        rmin_, rmax_ = np.quantile(values, 0.01), np.quantile(values, 0.99)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    for i, idx_i in enumerate(np.where(~drop_cols)[0]):
        if i >= max_rows:
            break
        for j, idx_j in enumerate(np.where(~drop_cols)[0]):
            if j >= max_cols:
                break
            ax = axes[i][j]

            if idx_i == idx_j:
                values = df.iloc[:, idx_i].values[mask[df.columns[idx_i]].values]

                # Deal with the diagonal by drawing a histogram there.
                if diagonal == "hist":
                    ax.hist(values, color=cmain, bins=20, **hist_kwds)

                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), color=cmain, **density_kwds)

                ax.set_xlim(boundaries_list[i])

            else:
                common = (mask[df.columns[idx_i]] & mask[df.columns[idx_j]]).values
                c = cmain

                kwargs_master = dict(
                    marker=marker, alpha=alpha,
                    color=c,
                    **kwds
                )
                if color_mask is not None and color_mask[idx_i, idx_j] == 0:
                    kwargs = {
                        **kwargs_master,
                        # "alpha": 0.2
                        "color": "lightgray"
                    }
                else:
                    kwargs = kwargs_master

                ax.scatter(
                    df.iloc[:, idx_j][common],  df.iloc[:, idx_i][common],
                    **kwargs
                )

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(df.columns[idx_j])
            ax.set_ylabel(df.columns[idx_i], rotation=0, labelpad=10)# fontsize=20, labelpad=20

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)

    if len(df.columns) > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)

    xlabelsize = 8
    xrot = 0
    ylabelsize = 8
    yrot = 0
    for ax in np.asarray(axes).ravel():
        if xlabelsize is not None:
            plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            plt.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            plt.setp(ax.get_yticklabels(), rotation=yrot)

    return axes



def visualize_data(x, g, size_per_var=1, max_cols=20, max_rows=100, show=False, sort_by_degree=False):
    assert x.ndim == 2
    # sort by degree
    if sort_by_degree:
        sorting = np.argsort(-g.sum(1))
        sorting_mat = np.eye(x.shape[-1])[:, sorting]

        x = x @ sorting_mat
        g = sorting_mat.T @ g @ sorting_mat

    # drop empty cols
    drop_cols = np.all(np.isclose(g, 0), axis=0) & np.all(np.isclose(g, 0), axis=1)

    # visualize in scatter matrix
    data = pd.DataFrame(x, columns=[r"$x_{" + f"{j}" + r"}$" for j in range(x.shape[-1])])
    _scatter_matrix(data, drop_cols=drop_cols, alpha=0.7, size_per_var=size_per_var, diagonal='hist',
                    max_cols=max_cols, max_rows=max_rows, color_mask=g)

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    if show:
        plt.show()
    log_img = wandb.Image(plt)
    plt.close()
    return log_img



def visualize_count_data(mat, max_rows=100, label_rows="cells", max_cols=30, label_cols="genes", size=0.5,
                         label_z="UMI", landscape=False, zsort="max", figsize=(5, 5), show=False):

    # adjust oversized inputs
    max_rows = min(max_rows, mat.shape[0])
    max_cols = min(max_cols, mat.shape[1])
    mat = mat[:max_rows, :max_cols]

    # set up figure

    # plot
    if landscape:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        _x = np.arange(max_cols)
        _y = np.arange(max_rows)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = np.flip(mat, axis=0).ravel()
        bottom = np.zeros_like(top)
        width = depth = 1

        # import matplotlib.colors as colors
        # import matplotlib.cm as cm
        # dz = top
        # offset = dz + np.abs(dz.min())
        # fracs = offset.astype(float) / offset.max()
        # norm = colors.Normalize(fracs.min(), fracs.max())
        # colors = cm.jet(norm(fracs.tolist()))
        colors = plt.cm.jet(top / top.max())
        ax.bar3d(x, y, bottom, width, depth, top, shade=True, zsort=zsort, color=colors)

        # ax.bar3d(x, y, bottom, width, depth, top, shade=True, cmap=cmap)
        # colorbar creation:
        # colorMap = plt.cm.ScalarMappable(cmap=plt.cm.rainbow_r)
        # colorMap.set_array(fourth_dim)
        # colBar = plt.colorbar(colorMap).set_label('fourth dimension')

        color_tuple = (1.0, 1.0, 1.0, 0.0)

        # background
        # ax.w_xaxis.set_pane_color(color_tuple)
        # ax.w_yaxis.set_pane_color(color_tuple)
        # ax.w_zaxis.set_pane_color(color_tuple)

        # axis lines
        ax.w_xaxis.line.set_color(color_tuple)
        ax.w_yaxis.line.set_color(color_tuple)
        ax.w_zaxis.line.set_color(color_tuple)

        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_zticks([])

        ax.set_xlabel(label_cols)
        ax.set_ylabel(label_rows)
        ax.set_zlabel(label_z)

        plt.tight_layout()

    else:
        fig = plt.figure(figsize=(max_cols * size, max_rows * size))
        ax = fig.add_subplot(111)

        maxval = np.max(mat)
        # ax.matshow(mat, vmin=0, vmax=maxval)
        im = ax.matshow(mat, cmap=plt.cm.plasma, norm=mc.LogNorm())

        # colorbar
        cbar = fig.colorbar(im, ax=ax, extend='max', shrink=0.5, aspect=30)
        cbar.minorticks_on()
        cbar.ax.set_ylabel('UMI count', rotation=90)

        # background color for 0 count
        ax.set_facecolor('whitesmoke')

        # axes
        # ax.set_title(f'title', pad=3, fontsize=30)
        ax.set_xlabel(label_cols)
        ax.set_ylabel(label_rows)

        ax.tick_params(axis='both', which='both', length=0)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        # hide axis spines/lines
        # for k in ax.spines.keys():
        #     ax.spines[k].set_visible(False)

        plt.tight_layout()


    descr = "3d" if landscape else "2d"
    plt.savefig(PROJECT_DIR / "plots" / f'count-mat-{descr}.pdf', format="pdf", dpi=100)
    if show:
        plt.show()
    plt.close()

    return




