import numpy as onp
import pprint as pprint
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats

from avici.definitions import *

from pprint import pprint
from collections import defaultdict

import pandas as pd


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = onp.linspace(0, 2*onp.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = onp.append(x, x[0])
                y = onp.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(onp.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta



COLOR_SATURATION = 0.8

DPI = 300

LINE_WIDTH = 7.0
COL_WIDTH = 3.333

FIG_SIZE_TRIPLE = (COL_WIDTH / 3, COL_WIDTH / 3 * 4/6)
FIG_SIZE_TRIPLE_TALL = (COL_WIDTH / 3, COL_WIDTH / 3 * 5/6)

FIG_SIZE_DOUBLE = (COL_WIDTH / 2, COL_WIDTH / 2 * 4/6)
FIG_SIZE_DOUBLE_TALL = (COL_WIDTH / 2, COL_WIDTH / 2 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, COL_WIDTH / 2 * 5/6)

FIG_SIZE_FULL_PAGE_TRIPLE = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 4/6)
FIG_SIZE_FULL_PAGE_TRIPLE_TALL = (LINE_WIDTH / 3, LINE_WIDTH / 3 * 5/6)

CUSTOM_FIG_SIZE_FULL_PAGE_QUAD = (LINE_WIDTH / 4, COL_WIDTH / 2 * 5/6)


NEURIPS_LINE_WIDTH = 5.5  # Text width: 5.5in (double figure minus spacing 0.2in).
FIG_SIZE_NEURIPS_DOUBLE = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 4/6)
FIG_SIZE_NEURIPS_TRIPLE = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 4/6)
FIG_SIZE_NEURIPS_DOUBLE_TALL = (NEURIPS_LINE_WIDTH / 2, NEURIPS_LINE_WIDTH / 2 * 5/6)
FIG_SIZE_NEURIPS_TRIPLE_TALL = (NEURIPS_LINE_WIDTH / 3, NEURIPS_LINE_WIDTH / 3 * 5/6)

#
COLORS_BRIGHT = [
    "#001AFD",  # 0 dark blue
    "#FC6608",  # 1 orange
    "#22C32B",  # 2 green
    "#DF000D",  # 3 red
    "#7600DA",  # 4 purple
    "#8C3703",  # 5 brown
    "#EB2BB4",  # 6 pink
    "#929292",  # 7 grey
    "#FEB910",  # 8 dark yellow
    "#1CCEFF",  # 9 light blue

]


# black -> blue -> white
COLORS_FLOW = [
    "#141414",  # 0
    "#1E273C",  # 1
    "#283D6F",  # 2
    "#3354A7",  # 3
    "#4F74CB",  # 4
    "#7F9AD7",  # 5
    "#B5C4E3",  # 6
    "#EFEFEF",  # 7
]


VARY_COLORS = [
    "#E3E116",
    "#2AA767",
    "#25567A",
    "#371460",
]


OOD_COLORS = [
    COLORS_BRIGHT[9],
    COLORS_BRIGHT[2],
    COLORS_BRIGHT[3],
    COLORS_BRIGHT[0],
]

OOD_ALPHAS = [
    0.2,
    0.1,
    0.1,
    0.1
]


PLOT_VARY = {
    "linear": [
        (10, "final/linear-generalization/d=10"),
        (20, "final/linear-generalization/d=20"),
        (50, "final/linear-generalization/d=50"),
        (100, "final/linear-generalization/d=100"),
    ],
    "rff": [
        (10, "final/rff-generalization/d=10"),
        (20, "final/rff-generalization/d=20"),
        (50, "final/rff-generalization/d=50"),
        (100, "final/rff-generalization/d=100"),
    ],
    "gene": [
        (10, "final/gene-generalization/d=10"),
        (20, "final/gene-generalization/d=20"),
        (50, "final/gene-generalization/d=50"),
        (100, "final/gene-generalization/d=100"),
    ],
}

PLOT_OOD = {
    "linear": [
        (r"in-distribution", "final/linear-generalization/in-dist"),
        (r"ood-graph", "final/linear-generalization/ood-graph"),
        (r"ood-graph-function", "final/linear-generalization/ood-graph-function"),
        (r"ood-graph-function-noise", "final/linear-generalization/ood-graph-function-noise"),
    ],
    "rff": [
        (r"in-distribution", "final/rff-generalization/in-dist"),
        (r"ood-graph", "final/rff-generalization/ood-graph"),
        (r"ood-graph-function", "final/rff-generalization/ood-graph-function"),
        (r"ood-graph-function-noise", "final/rff-generalization/ood-graph-function-noise"),
    ],
    "gene": [
        (r"in-distribution", "final/gene-generalization/in-dist"),
        (r"ood-graph", "final/gene-generalization/ood-graph"),
        (r"ood-graph-function", "final/gene-generalization/ood-graph-function"),
        (r"ood-graph-function-noise", "final/gene-generalization/ood-graph-function-noise"),
    ],
}


def row_to_value(row, key):
    if key == "method":
        base_method = row["method"].split("__")[0]
        return base_method
    else:
        base_method, spec = row["method"].split("__", 1)
        spl = spec.split("=")
        assert len(spl) == 2
        return int(spl[1])

metric_label_dict = {
    "f1": "F1",
    "precision": "Prec.",
    "recall": "Rec.",
    "auprc": "AUPRC",
    "auroc": "AUROC",
    "acyc": r"   $\%$ acyclic",
    "cyclic": r" $\%$ cyclic",
    "shd": r"SHD",
    "sid": r"SID",
    "shd_ratio": r"    SHD${}_{ratio}$", # spaces are needed to shift out
    "sid_ratio": r"SID${}_{ratio}$",
}

not_zero_one_metrics = {
    "sid",
    "shd",
}

ood_setting_label_dict = {
    "in-distribution": "in-dist.",
    "ood-graph": "o.o.d. $G$",
    "ood-graph-function": "o.o.d. $G$, $f$",
    "ood-graph-function-noise": "o.o.d. $G$, $f$, $\epsilon$",
}

ood_gene_setting_label_dict = {
    "in-distribution": "in-dist.",
    "ood-graph": "o.o.d. $G$",
    "ood-graph-function": "o.o.d. $G$, sim",
    "ood-graph-function-noise": "o.o.d. $G$, sim, noise",
}


def quantile_lower(x):
    return onp.percentile(x, 25)
quantile_lower.__name__ = "quantile_lower"

def quantile_upper(x):
    return onp.percentile(x, 75)
quantile_upper.__name__ = "quantile_upper"


if __name__ == '__main__':

    vary_figsize = (3., 3.)
    # vary_plot_metrics = ["f1", "auprc", "shd", "sid", "cyclic"]
    vary_plot_metrics = ["f1", "auprc"]
    vary_plot_grid_name = r"$d$"
    vary_method = "ours-interv"
    # vary_agg_metrics = [onp.mean, onp.std]
    vary_agg_metrics = [onp.mean, quantile_lower, quantile_upper]
    generate_plot_vary = True


    ood_frame_style = "polygon"
    # ood_frame_style = "circle"
    ood_figsize = (3., 3.)
    ood_radar_ring_vals = [0.0, 0.2, 0.4, 0.6, 0.8]
    ring_vals_angle = 200
    ood_ylim = (0.0, 1.0)
    ood_method = "ours-interv"
    ood_metrics = ["shd", "sid", "f1", "precision", "recall", "auprc", "auroc", "acyc"]

    ood_title_dict = {
        "linear": "Linear",
        "rff": "RFF",
        "gene": "GRN",
    }
    generate_plot_ood = True

    # plt.rcParams.update(NEURIPS_RCPARAMS)

    # complete paths with loaded summary dfs
    results_folder = PROJECT_DIR / RESULTS_SUBDIR
    for parse_row, dictt in [(False, PLOT_OOD), (True, PLOT_VARY)]:
        for k, setting_spec in dictt.items():

            # for each summary folder, take latest summary and load df
            new_dictt_k = []
            for name, p in [(name, results_folder / p) for name, p in setting_spec]:

                summaries = sorted([subp for subp in p.iterdir() if "summary" in subp.name])
                filtered_summaries = list(filter(lambda p: p.name.rsplit("_", 1)[-1] == "final", summaries))
                if filtered_summaries:
                    assert len(filtered_summaries) == 1
                    last_summary = filtered_summaries[0]
                else:
                    last_summary = summaries[-1]

                # load df
                df = pd.read_csv(index_col=0, filepath_or_buffer=(last_summary / "df").with_suffix(".csv"))

                # convert names into new categories columns (e.g. lim n_obs)
                if parse_row:
                    df["lim_n_obs"] = df.apply(lambda row: row_to_value(row, "lim_n_obs"), axis=1)

                # lastly, update name
                df["method"] = df.apply(lambda row: row_to_value(row, "method"), axis=1)

                #
                new_dictt_k.append((name, df))

            dictt[k] = new_dictt_k


    # =======================================================================
    # VARY plot
    if generate_plot_vary:

        for plot_name, setting_spec in PLOT_VARY.items():

            for vary_plot_metr in vary_plot_metrics:

                # query plot data
                plot_data = defaultdict(dict)
                for n_vars, df in setting_spec:

                    # filter desired metric
                    df = df.loc[df["metric"] == vary_plot_metr].drop("metric", axis=1)

                    # filter desired method
                    df = df.loc[df["method"] == vary_method]
                    assert len(df["method"].unique()) == 1, f"{plot_name} {vary_plot_metr} {str(df['method'].unique())}"
                    df = df.drop("method", axis=1)

                    # summarize df metric
                    table = df.pivot_table(
                        columns=["lim_n_obs"],
                        values="val",
                        aggfunc=vary_agg_metrics) \
                        .swaplevel(-1, -2, axis=1) \
                        .sort_index()

                    lim_n_obs_vals = list(sorted(df["lim_n_obs"].unique()))
                    for n_obs in lim_n_obs_vals:
                        assert n_vars not in plot_data[n_obs]
                        # plot_data[n_obs][n_vars] = (table[(n_obs, "mean")].item(), table[(n_obs, "std")].item())
                        plot_data[n_obs][n_vars] = (table[(n_obs, "mean")].item(),
                                                    table[(n_obs, "quantile_lower")].item(),
                                                    table[(n_obs, "quantile_upper")].item())

                # plot
                fig, ax = plt.subplots(1, 1, figsize=vary_figsize)
                yax_offs = 5
                yax_inc = 1
                n_vars_reversed = True
                n_obs_ordered = sorted(plot_data.keys())
                for i, n_obs in enumerate(n_obs_ordered):
                    n_obs_dict = plot_data[n_obs]
                    n_vars_list = list(sorted(n_obs_dict.keys(), reverse=n_vars_reversed))
                    # vals, errs = zip(*[n_obs_dict[d] for d in n_vars_list])
                    vals, val_lower_bound, val_upper_bound = zip(*[n_obs_dict[d] for d in n_vars_list])

                    # compute gap between mean and lower/upper percentile
                    err_lo = onp.maximum(onp.array(vals) - onp.array(val_lower_bound), 0.0)
                    err_hi = onp.maximum(onp.array(val_upper_bound) - onp.array(vals), 0.0)
                    errs = onp.vstack([err_lo, err_hi])

                    yax_pos = yax_inc * (len(n_obs_ordered) - i - 1)
                    ys = [yax_pos + hh * yax_offs for hh in range(len(n_vars_list))]

                    ax.errorbar(x=vals, y=ys, xerr=errs, marker="o", color=VARY_COLORS[i],
                                linestyle="none", capthick=1, label=rf"$n$={n_obs}", capsize=3)

                ylabel_pos = [(len(plot_data) - 1)/2 + hh * yax_offs for hh in range(len(n_vars_list))]
                ax.yaxis.set_major_locator(plt.FixedLocator(ylabel_pos))

                def format_func(tick_val, tick_number):
                    return n_vars_list[tick_number]

                ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

                if vary_plot_metr not in not_zero_one_metrics:
                    ax.set_xlim((0, 1))

                ax.set_xlabel(metric_label_dict[vary_plot_metr])
                ax.set_ylabel(vary_plot_grid_name, rotation=0) #  fontsize=20, labelpad=20

                fig.legend(loc="lower left", ncol=2, labelspacing=0.03,
                           bbox_to_anchor=[0.2, 0.0], fontsize='small')
                # legend = ax.legend(labels, loc=(0.75, .95), labelspacing=0.1, )
                fig.tight_layout(rect=[0.0, 0.1, 1, 1]) # default 0, 0, 1, 1


                plt.savefig((results_folder.parent / "paper_plots" / f"scaling_{plot_name}_{vary_plot_metr}")\
                            .with_suffix(".pdf"), format="pdf", facecolor=None,
                            dpi=DPI, bbox_inches='tight')

                plt.show()


    # =======================================================================
    # VARY plot
    if generate_plot_ood:

        for plot_name, setting_spec in PLOT_OOD.items():

            # query plot data
            plot_data_ood = defaultdict(dict)
            for ood_setting, df in setting_spec:

                # filter desired method
                df = df.loc[df["method"] == ood_method]
                assert len(df["method"].unique()) == 1
                df = df.drop("method", axis=1)

                # summarize df metric
                table = df.pivot_table(
                    columns=["metric"],
                    values="val",
                    aggfunc=[onp.mean]) \
                    .swaplevel(-1, -2, axis=1) \
                    .sort_index()\
                    .droplevel(1, axis=1)

                # convert cyclicity into acyclicity
                table["acyc"] = 1 - table["cyclic"]
                table = table.drop("cyclic", axis=1)

                # lim_n_obs_vals = list(sorted(df["lim_n_obs"].unique()))
                for metr in ood_metrics:
                    assert metr not in plot_data_ood[ood_setting], f"{metr} {list(plot_data_ood[ood_setting])}"
                    if metr in table.columns:
                        plot_data_ood[ood_setting][metr] = table[metr].item()

            # convert SHD/SID to ratio SHD_in-dist/SHD
            for ratio_metric in ["shd", "sid"]:
                if ratio_metric in plot_data_ood["in-distribution"]:
                    for ood_setting in plot_data_ood.keys():
                        plot_data_ood[ood_setting][f"{ratio_metric}_ratio"] = \
                            plot_data_ood["in-distribution"][ratio_metric] / plot_data_ood[ood_setting][ratio_metric]
                    for ood_setting in plot_data_ood.keys():
                        del plot_data_ood[ood_setting][ratio_metric]


            available_metrics = list(plot_data_ood["in-distribution"].keys())

            # plot
            labels = []
            ood_data_radar = []
            for ood_setting, _ in setting_spec:
                labels.append((ood_setting_label_dict if plot_name != "gene" else ood_gene_setting_label_dict)[ood_setting])
                # labels.append(ood_setting_label_dict[ood_setting])
                ood_data_radar.append([plot_data_ood[ood_setting][metr] for metr in available_metrics])

            theta = radar_factory(len(available_metrics), frame=ood_frame_style)
            fig, ax = plt.subplots(figsize=ood_figsize, nrows=1, ncols=1, subplot_kw=dict(projection='radar'))
            # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
            fig.subplots_adjust(bottom=0.26, left=-0.02, right=0.98)

            # Plot the four cases from the example data on separate axes
            ax.set_rgrids(ood_radar_ring_vals, angle=ring_vals_angle)
            # ax.set_title(ood_title_dict[plot_name], weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')

            spoke_labels = [metric_label_dict[metr] for metr in available_metrics]
            for d, color, alpha in zip(ood_data_radar, OOD_COLORS, OOD_ALPHAS):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=alpha)
            ax.set_varlabels(spoke_labels)
            ax.set_ylim(ood_ylim)


            # legend = ax.legend(labels, loc=(0.75, .95), labelspacing=0.1, fontsize='small')
            # legend = ax.legend(labels, loc=(0.0, 0.0), labelspacing=0.1, fontsize='small', )
            # legend = ax.legend(labels, bbox_to_anchor=(0.5, 0.0), labelspacing=0.1, fontsize='small', ncol=2)
            # legend = fig.legend(labels, loc=(0.75, .95), labelspacing=0.1, fontsize='small', ncol=2)
            # legend = fig.legend(labels, bbox_to_anchor=(0.75, .95), labelspacing=0.1, fontsize='small', ncol=2)

            # if plot_name != "gene":
            #     legend = fig.legend(labels, bbox_to_anchor=(0.95, 0.17), labelspacing=0.03, fontsize='small', ncol=2)
            # else:
            #     legend = fig.legend(labels, bbox_to_anchor=(1.01, 0.17), labelspacing=0.03, fontsize='small', ncol=2)

            if plot_name != "gene":
                legend = fig.legend(labels, bbox_to_anchor=(0.85, 0.17), labelspacing=0.03, fontsize='small', ncol=2)
            else:
                legend = fig.legend(labels, bbox_to_anchor=(0.95, 0.17), labelspacing=0.03, fontsize='small', ncol=2)

            # plt.tight_layout()


            plt.savefig((results_folder.parent / "paper_plots" / f"radar_ood_{plot_name}").with_suffix(".pdf"), format="pdf", facecolor=None,
                        dpi=DPI, bbox_inches='tight')


            plt.show()
