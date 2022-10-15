import numpy as onp
import pprint as pprint
import matplotlib.pyplot as plt

from avici.utils.rff import make_rff_rbf_function

from avici.synthetic.graph import *
from avici.definitions import *
import scipy.stats as stats

DPI = 300
save_folder = PROJECT_DIR / "paper_plots"
noise_color = "#001AFD"

if __name__ == '__main__':

    plot_graph = True
    plot_functions = True
    plot_noise = True

    save_plots = True

    # plot_graph = False
    # plot_functions = False
    # plot_noise = True
    #
    # save_plots = False

    figsize = (2.5, 2.5)
    xlim = (-7, 7)
    ylim = (-20, 20)
    noise_ylim = (-7, 7)
    n_examples = 1 # to g generate the second rff plot, set this to 2

    # graph
    d = 30
    permute = False

    # function
    # length scale, output scale
    function_settings = [
        (8, 15),
        (5, 20),
    ]
    function_view_angle = dict(elev=20, azim=60)

    # noise
    noise_length_scale = 10.0
    noise_output_scale = 2.0
    n_noise_samples = 100


    ######
    if plot_graph:
        rng = onp.random.default_rng(2)
        # graph
        for name, model, params in [
            (r"Erdos-Renyi", erdos_renyi, dict(edges_per_var=2, permute=permute, flip=True)),
            (r"Scale-free", scale_free, dict(edges_per_var=2, power=1, permute=permute)),
            (r"Watts-Strogatz", watts_strogatz, dict(dim=2, nei=1, p=0.2, permute=permute)),
            (r"Stochastic Block Model", sbm, dict(edges_per_var=2, n_blocks=5, damp=0.1, permute=permute)),
            (r"Geometric Random", grg, dict(radius=0.10, permute=permute)),
        ]:

            for idx in range(n_examples):
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                g = model(rng, d, **params)[0]
                n_edges = g.sum()
                ax.set_title(name)
                ax.matshow(g, vmin=0, vmax=1)
                ax.tick_params(axis='both', which='both', length=0)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.axis('off')
                plt.tight_layout()


                if save_plots:
                    savename = name.replace(" ", "_").lower()
                    plt.savefig((save_folder / f"example_graph_{savename}_{idx}").with_suffix(".pdf"), format="pdf", facecolor=None,
                                dpi=DPI, bbox_inches='tight')
                else:
                    plt.show()

    # functions
    N = 100
    if plot_functions:
        rng = onp.random.default_rng(3)
        x_star = onp.linspace(*xlim, N).reshape(N, 1)
        for length_sc, output_sc in function_settings:
            # 3d sample
            for idx in range(n_examples):
                xx, yy = onp.meshgrid(x_star.flatten(), x_star.flatten())
                xy = onp.stack([xx, yy]).reshape(2, -1).T

                # rff
                z = make_rff_rbf_function(rng=rng, d=2, c=output_sc, ls=length_sc, n_rff=100)(xy)

                fig = plt.figure(figsize=figsize)
                ax = plt.axes(projection='3d')
                # ax.contour3D(xx, yy, z.reshape(gp.N, gp.N), 50, cmap='binary')
                ax.plot_surface(xx, yy, z.reshape(N, N), rstride=1, cstride=1,
                                cmap='viridis', edgecolor='none')
                # ax.view_init(60, 35)
                ax.view_init(**function_view_angle) # view angle
                ax.set_xlabel(r'$x_i$')
                ax.set_ylabel(r'$x_j$')
                # ax.set_zlabel(r'$f$')
                ax.zaxis.set_rotate_label(False)  # disable automatic rotation
                ax.set_zlabel(r"$f$", rotation=0)
                ax.set_zlim(ylim)
                plt.title(rf"$c = {output_sc}, \ell = {length_sc}$")
                plt.tight_layout(rect=[0.15, 0.05, 1.0, 1.0])
                if save_plots:
                    plt.savefig((save_folder / f"example_rff_{idx}_ls={length_sc}_c={output_sc}").with_suffix(".pdf"), format="pdf", facecolor=None,
                                dpi=DPI, bbox_inches='tight')
                else:
                    plt.show()


    # noise
    if plot_noise:
        rng = onp.random.default_rng(0)


        def squash(arr):
            return onp.log(1 + onp.exp(arr))

        contours = {}
        for perc in [0.66, 0.95, 0.999]:
            tail = (1.0 - perc) / 2.0
            contours[perc] = {}
            for name, dist in [("gaussian", stats.norm), ("laplace", stats.laplace), ("cauchy", stats.cauchy)]:
                lo, hi = dist.ppf([tail, 1.0 - tail])
                contours[perc][name] = lo, hi

        # gaussian
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        locs = rng.uniform(*xlim, size=n_noise_samples)
        locs_even = onp.linspace(*xlim, N)
        ax.scatter(locs, rng.normal(0, size=n_noise_samples), marker="x", linewidth=1, c="black", alpha=0.5)
        ax.plot(locs_even, 0.0 * onp.ones_like(locs_even), linewidth=2, color=noise_color)
        for _, dd in contours.items():
            lo, hi = dd["gaussian"]
            ax.plot(locs_even, lo * onp.ones_like(locs_even), linewidth=1, color=noise_color, linestyle="dashed")
            ax.plot(locs_even, hi * onp.ones_like(locs_even), linewidth=1, color=noise_color, linestyle="dashed")

        ax.set_ylim(noise_ylim)
        ax.set_xlabel(rf"$x_j$")
        ax.set_ylabel(rf"$\epsilon_j$", rotation=0)  # fontsize=20, labelpad=20
        ax.set_title(r"N$(0,1)$")
        plt.tight_layout()
        if save_plots:
            plt.savefig((save_folder / f"example_noise_gaussian").with_suffix(".pdf"), format="pdf",
                        facecolor=None,
                        dpi=DPI, bbox_inches='tight')
        else:
            plt.show()


        # 3d sample
        for idx in range(n_examples):

            # laplace
            fig, ax = plt.subplots(1, 1, figsize=figsize)

            x_star = onp.linspace(*xlim, N).reshape(N, 1)
            xx, yy = onp.meshgrid(x_star.flatten(), x_star.flatten())
            xy_stack = onp.stack([xx, yy])
            xy = xy_stack.reshape(2, -1).T

            # rff
            f = make_rff_rbf_function(rng=rng, d=2, c=noise_output_scale, ls=noise_length_scale, n_rff=100)
            z = f(xy)

            # plot noise samples for two dim slices at 0
            locs = rng.uniform(*xlim, size=n_noise_samples)
            locs_even = onp.linspace(*xlim, N)

            xy_slice = onp.vstack([locs, onp.zeros_like(locs)]).T
            xy_slice_even = onp.vstack([locs_even, onp.zeros_like(locs_even)]).T

            stddev_slice = onp.sqrt(squash(f(xy_slice)))
            stddev_slice_even = onp.sqrt(squash(f(xy_slice_even)))

            samples = rng.laplace(0, stddev_slice)

            ax.scatter(xy_slice[:, 0], samples, marker="x", linewidth=1, c="black", alpha=0.5)
            ax.plot(locs_even, 0.0 * onp.ones_like(locs_even), linewidth=2, color=noise_color)
            for _, dd in contours.items():
                lo, hi = dd["laplace"]
                ax.plot(locs_even, lo * stddev_slice_even, linewidth=1, color=noise_color, linestyle="dashed")
                ax.plot(locs_even, hi * stddev_slice_even, linewidth=1, color=noise_color, linestyle="dashed")

            ax.set_ylim(noise_ylim)
            ax.set_xlabel(rf"$x_j$")
            ax.set_ylabel(rf"$\epsilon_j$",rotation=0)  # fontsize=20, labelpad=20
            ax.set_title(r"Laplace$(0,\sigma^2)$, $\sigma = h_j(x_{pa(j)})$")

            if save_plots:
                plt.savefig((save_folder / f"example_noise_laplace_{idx}").with_suffix(".pdf"), format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')
            else:
                plt.show()