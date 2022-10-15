import warnings
warnings.filterwarnings("ignore", message="No GPU automatically detected")
warnings.formatwarning = lambda msg, category, path, lineno, file: f"{path}:{lineno}: {category.__name__}: {msg}\n"

import itertools
import matplotlib
import matplotlib.transforms
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.colors import ListedColormap
import matplotlib.font_manager # avoids loading error of fontfamily["serif"]

import seaborn as sns
import numpy as onp
import scipy
import pandas as pd
from collections import defaultdict
from pprint import pprint

from avici.eval.metrics import make_calibration_stats


from avici.definitions import BASELINE_ZERO, BASELINE_RAND, BASELINE_RAND_EDGES, BASELINES_INTERV, BASELINES_OBSERV

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

NEURIPS_RCPARAMS = {
    "figure.autolayout": True,       # `False` makes `fig.tight_layout()` not work
    "figure.figsize": FIG_SIZE_NEURIPS_DOUBLE,
    # "figure.dpi": DPI,             # messes up figisize
    # Axes params
    "axes.linewidth": 0.5,           # Matplotlib's current default is 0.8.
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,

    "hatch.linewidth": 0.3,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    'xtick.major.pad': 3.0,
    'xtick.major.size': 1.75,
    'xtick.minor.pad': 1.0,
    'xtick.minor.size': 1.0,

    'ytick.major.pad': 1.0,
    'ytick.major.size': 1.75,
    'ytick.minor.pad': 1.0,
    'ytick.minor.size': 1.0,

    "axes.labelpad": 0.5,
    # Grid
    "grid.linewidth": 0.3,
    # Plot params
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    'errorbar.capsize': 3.0,
    # Font
    # "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",             # use serif rather than sans-serif
    "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
    "font.size": 8.5,
    "axes.titlesize": 8.5,                # LaTeX default is 10pt font.
    "axes.labelsize": 8.5,                # LaTeX default is 10pt font.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    # Legend
    "legend.fontsize": 7,        # Make the legend/label fonts a little smaller
    "legend.frameon": True,              # Remove the black frame around the legend
    "legend.handletextpad": 0.3,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.1,
    "patch.linewidth": 0.5,
    # PDF
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
    "pgf.preamble": (
        r'\usepackage{fontspec}'
        r'\usepackage{unicode-math}'
        r'\setmainfont{Times New Roman}'
    ),
    "text.latex.preamble": (
        r'\usepackage{amsmath}'
        r'\usepackage{amssymb}'
        r'\usepackage{amsfonts}'
    ),
}

COLORS_0 = [
    "#DF000D", # red
    "#23DD20", # green
    "#1ACBF7", # light blue
    "#FB0D89", # pink
    "#7600DA", # purple
    "#FC6608", # orange
    "#001AFD", # dark blue
]

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

COLORS = COLORS_BRIGHT

METHODS_CONFIG = [
    # observ
    ("ges",              COLORS[0], False, "GES"),
    ("bootstrap-ges",    COLORS[0], False, r"GES${}^*$"),

    ("lingam",           COLORS[1], False, "LiNGAM"),
    ("bootstrap-lingam", COLORS[1], False, r"LiNGAM${}^*$"),

    ("pc",               COLORS[2], False, "PC"),
    ("bootstrap-pc",     COLORS[2], False, r"PC${}^*$"),

    ("daggnn",           COLORS[9], False, "DAG-GNN"),
    ("bootstrap-daggnn", COLORS[9], False, r"DAG-GNN${}^*$"),

    ("grandag",          COLORS[4], False, "GraN-DAG"),
    ("bootstrap-grandag",COLORS[4], False, r"GraN-DAG${}^*$"),

    ("ours-observ",      COLORS[3], False, "AVICI (obs)"),

    # interv
    ("gies",             COLORS[5], True, "GIES"),
    ("bootstrap-gies",   COLORS[5], True, r"GIES${}^*$"),

    ("igsp",             COLORS[6], True, "IGSP"),
    ("bootstrap-igsp",   COLORS[6], True, r"IGSP${}^*$"),

    ("dcdi",             COLORS[7], True, "DCDI"),
    ("bootstrap-dcdi",   COLORS[7], True, r"DCDI${}^*$"),

    ("dibs",             COLORS[8], True, "DiBS"),

    ("ours-interv",      COLORS[3], True, "AVICI"),
]
METHODS_DEFAULT_CONFIG = (COLORS[-1], "default")


METHODS_CONFIG_CALIBRATION = METHODS_CONFIG

CALIBRATION_PLOT_COLOR_SINGLE = COLORS[4]

TRIVIAL_CONFIG = [
    (BASELINE_ZERO,        ("#000000", "solid",  1.5)),
    (BASELINE_RAND,        ("#000000", "dashed", 1.5)),
    (BASELINE_RAND_EDGES,  ("#000000", "dotted", 1.5)),
]

TRIVIAL_METHODS = [k for k, _ in TRIVIAL_CONFIG]

CALIBRATION_XLABEL = r"predicted $\hat{p}(g_{i,j}=1\,|\,D)$"
CALIBRATION_YLABEL = r"empirical $p(g_{i,j}=1\,|\,D)$"

METRICS_TABLE_ORDERING = [
    "heldout_score",
    "shd",
    "sid",
    "f1",
    "precision",
    "recall",
    "holl",
    "auroc",
    "auprc",
    "ap",
    "ece",
    "brier",
]

# rest round to 1 decimal
METRICS_HIGHER_PREC = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "ap",
    "ece",
    "brier",
]


# metrics
METRICS_HIGHER_BETTER = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "ap",
    "heldout_score",
    "pair-precision",
    "pair-recall",
    "pair-f1",
    "pair-auroc",
    "pair-auprc",
    "pair-ap",
]
METRICS_LOWER_BETTER = [
    "sid",
    "shd",
    "ece",
    "brier",
    "walltime",
    "cyclic",
]


def _add_t_test_markers(df, table, best_aggmetr):
    # mark best methods
    table["_obs"] = [meth.rsplit("bootstrap-", 1)[-1].split("__", 1)[0] in BASELINES_OBSERV for meth in table.index]
    table["_int"] = [meth.rsplit("bootstrap-", 1)[-1].split("__", 1)[0] in BASELINES_INTERV for meth in table.index]
    assert all(table["_obs"] ^ table["_int"]), f"Not all methods have observ/interv status: {table.index}"
    method_categories = list(filter(lambda c: table[c].sum() > 0, ["_obs", "_int"]))

    for metr in df.metric.unique():
        for method_category in method_categories:
            methods_sel = table[method_category]
            if not methods_sel.sum():
                continue

            lo = (table.loc[methods_sel, (metr, best_aggmetr)] ==
                  table.loc[methods_sel, (metr, best_aggmetr)].min())
            hi = (table.loc[methods_sel, (metr, best_aggmetr)] ==
                  table.loc[methods_sel, (metr, best_aggmetr)].max())

            assert (metr in METRICS_LOWER_BETTER) != (metr in METRICS_HIGHER_BETTER)
            table[(metr, "best" + method_category)] = lo if metr in METRICS_LOWER_BETTER else hi
            assert table[(metr, "best" + method_category)].sum() == 1, f'\n{table[(metr, "best" + method_category)]}'

    table = table.sort_index(axis="columns", level="metric")

    # do t-test (unequal variances, two-sided) with best method (per metric and per category)
    for metr in df.metric.unique():
        for method_category in method_categories:
            best_method = table.loc[table[(metr, "best" + method_category)] == True].index.values[0]
            best_values = df.loc[(df.method == best_method) & (df.metric == metr)].val.values
            methods_sel = table[method_category]

            ttest_func = lambda arr: scipy.stats.ttest_ind(arr, best_values, equal_var=False, alternative="two-sided")[1]
            pvals = df.pivot_table(
                index=["method"],
                columns="metric",
                values="val",
                aggfunc=ttest_func,
                dropna=False)\
                .loc[methods_sel, metr]

            table[(metr, "pval_best" + method_category)] = pvals

    # add an indicator for methods to be marked
    for metr in df.metric.unique():
        table[(metr, "_marked_")] = onp.zeros_like(table.index).astype(bool)
        for method_category in method_categories:
            # mark best
            table.at[(table[(metr, "best" + method_category)] == True), (metr, "_marked_")] = True

            # mark for p-val > 0.05 (inside 95% confidence interval of t-distribution)
            table.at[(table[(metr, "pval_best" + method_category)] >= 0.05), (metr, "_marked_")] = True

            table = table.drop([(metr, "best" + method_category),
                                (metr, "pval_best" + method_category)], axis=1, errors="ignore")

    table = table.drop(["_obs", "_int"], axis=1, errors="ignore").sort_index(axis="columns", level="metric")

    return table


def _format_table_str(full_table, metr, metrname):

    # format str
    mean_str_formatter = (lambda val: f"{val:.3f}") if metr in METRICS_HIGHER_PREC else (lambda val: f"{val:.1f}")
    add_str_formatter = (lambda val: " \\err{" + f"{val:.2f}" + "}") \
        if metr in METRICS_HIGHER_PREC else (lambda val: " \\err{" + f"{val:.1f}" + "}")

    rounding = 3 if metr in METRICS_HIGHER_PREC else 1
    add_rounding = 2 if metr in METRICS_HIGHER_PREC else 1

    full_table.loc[:, (metr, "str")] = \
        full_table.loc[:, (metr, metrname[0])].round(rounding).apply(mean_str_formatter) + \
        full_table.loc[:, (metr, metrname[1])].round(add_rounding).apply(add_str_formatter)

    full_table.drop(columns=(metr, metrname[0]), inplace=True)
    full_table.drop(columns=(metr, metrname[1]), inplace=True)

    # add marker str
    marked = "\\highlight{" + full_table.loc[:, (metr, "str")] + "}"
    is_marked = full_table.loc[:, (metr, "_marked_")]
    full_table.loc[is_marked, (metr, "str")] = marked.loc[is_marked]
    full_table.drop(columns=(metr, "_marked_"), inplace=True)
    return full_table


def benchmark_summary(save_path, method_results_input, only_metrics, ax_width=2.0, ax_height=4.0, show_trivial=False,
                      method_summaries=False, dump_main=False):

    if not dump_main:
        print("\nbenchmark_summary:", only_metrics if not None else "<all metrics>")
    assert not (dump_main and only_metrics is not None), "should only dump the full df"

    # preprocess results
    metric_results_dict = defaultdict(dict)
    for method, d in method_results_input.items():
        for metr, l in d.items():
            if only_metrics is not None and metr not in only_metrics:
                continue
            metric_results_dict[metr][method] = l
    if only_metrics is not None:
        metric_results = [(metr, metric_results_dict[metr]) for metr in only_metrics if metric_results_dict[metr]] # to order
    else:
        metric_results = sorted(list(metric_results_dict.items()))

    # create long list of (metric, method, value) tuples which are then aggregated into a df
    df = []
    for metric, res in metric_results:
        for m, l in res.items():
            for v in l:
                df.append((metric, m, v))
    df = pd.DataFrame(df, columns=["metric", "method", "val"])

    if df.empty:
        warnings.warn(f"\nNo results reported for metrics `{only_metrics}`. "
                      f"Methods: `{list(method_results_input.keys())}` ")
        return

    # dump summary table of metrics
    save_path.parent.mkdir(exist_ok=True, parents=True)
    if dump_main:
        df.to_csv(path_or_buf=(save_path.parent / "df").with_suffix(".csv"))
        print("dumped summary df")
        return

    # check whether we need to aggregate at all (not for real data, for which we only measure 1 metric score per method)
    is_singular = all([df.loc[(df.metric == metr) & (df.method == meth)].val.size == 1
        for metr, meth in itertools.product(df.metric.unique(), df.method.unique())])
    if is_singular:
        print("*" * 30 + "\nWarning: singular measurements; ignore std dev, std err or t-tests.")
        df = pd.concat([df, df]).reset_index(drop=True)

    # create table
    table_agg_metrics = [onp.mean, scipy.stats.sem]
    metrname = ["mean", "sem"]
    best_aggmetr = "mean"
    table = df.pivot_table(
        index=["method"],
        columns=["metric"],
        values="val",
        aggfunc=table_agg_metrics,
        dropna=False) \
        .drop(TRIVIAL_METHODS, axis=0, errors="ignore")\
        .swaplevel(-1, -2, axis=1)\
        .sort_index(axis="columns", level="metric")\
        .sort_index()

    # add marker for best
    table = _add_t_test_markers(df, table, best_aggmetr)

    # reorder and rename rows
    method_order = [method for method, *_ in METHODS_CONFIG]
    for method in sorted(table.index):
        if method not in method_order:
            method_order.append(method)
    all_methods_sorted = [method for method in method_order if method in table.index]
    full_table = table.reindex(all_methods_sorted)

    # convert full table to string in correct float format
    # mean (sem)
    # print(full_table)
    # print(full_table.columns.values)
    # print(full_table.columns.names)
    # print(full_table.columns)
    # print()
    # print(full_table.loc[:, ("f1", "mean")])
    # print(full_table.loc[:, ("f1", "mean")].round(3).apply(lambda val: f"{val:.3f}"))
    # print(full_table.loc[:, ("f1", "mean")].round(3).apply(lambda val: f"{val:.3f}")
    #       + full_table.loc[:, ("shd", "mean")].round(3).apply(lambda val: f" ({val:.3f})"))

    all_metrics = set(m for m, _ in full_table.columns.values)
    all_metrics_ordered = [metric for metric in METRICS_TABLE_ORDERING if metric in all_metrics]
    for metr in all_metrics_ordered:
        full_table = _format_table_str(full_table, metr, metrname)

    full_table = full_table.droplevel(1, axis=1) # drop the dummy string level

    if method_summaries:
        # save individual table for each method
        save_path_methods = save_path.parent / "methods"
        save_path_methods.mkdir(exist_ok=True, parents=True)
        base_methods = set(meth.split("__")[0] for meth in table.index if meth not in TRIVIAL_METHODS)
        for base_method in base_methods:
            method_table = table[[base_method in s for s in table.index]]
            heldout_score_exists = "heldout_score" in method_table.columns.levels[0] and \
                                   not method_table[('heldout_score', 'mean')].isnull().all()
            sort_crit = "heldout_score" if heldout_score_exists else ("sid" if "sid" in only_metrics else only_metrics[0])
            method_table = method_table.reindex(method_table[sort_crit].sort_values(by="mean", ascending=True).index)
            method_table.to_csv(path_or_buf=(save_path_methods / base_method).with_suffix(".csv"))
        return

    print(full_table)

    full_table.to_csv(path_or_buf=save_path.with_suffix(".csv"))
    full_table.to_latex(buf=save_path.with_suffix(".tex"), escape=False)

    # set plot configuration
    # sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS) 
    fig, axs = plt.subplots(1, len(metric_results), figsize=(ax_width * len(metric_results), ax_height))
    handles, labels = [], []

    for ii, (metric, _) in enumerate(metric_results):
        df_metric = df.loc[df["metric"] == metric].drop("metric", axis=1)

        # add placeholder if method not measured for this metric
        for m in method_results_input.keys():
            if m not in df_metric["method"].unique():
                df_metric = df_metric.append({"method": m, "val": onp.nan}, ignore_index=True)

        # get trivial baselines and drop from main plot
        trivial = []
        for k in TRIVIAL_METHODS:
            k_vals = df_metric.loc[df_metric["method"] == k].drop("method", axis=1)
            if not k_vals.isnull().values.all():
                trivial.append((k, k_vals))
            df_metric = df_metric.loc[df_metric["method"] != k]
        df_metric = df_metric.reset_index(drop=True)

        # formatting of methods
        unique = df_metric["method"].unique()
        config = []
        seen = set()
        for k, *tup in METHODS_CONFIG:
            matches = list(filter(lambda w: k == w.split("__")[0], unique))
            if matches:
                assert not any([w in seen for w in matches]), f"matches `{matches}` already seen"
                seen.update(matches)
                for w in matches:
                    config.append((w, tup[0]))

        for w in filter(lambda w: w not in seen, unique):
            # unknown method substring
            config.append((w, METHODS_DEFAULT_CONFIG[0]))

        # plot
        plot_kwargs = dict(
            ax=axs[ii],
            x="method",
            y="val",
            hue="method",
            dodge=False,  # violin positions do not change position or width according to hue
            data=df_metric,
            order=list(w for w, *_ in config),
            hue_order=list(w for w, *_ in config),
            palette=dict(config),
            saturation=COLOR_SATURATION,
        )

        # sns.violinplot(**plot_kwargs)
        # sns.boxenplot(**plot_kwargs)
        obj = sns.boxplot(**plot_kwargs)

        ymin_methods, ymax_methods = axs[ii].get_ylim()

        # add horizontal lines for trivial baselines
        if show_trivial:
            for k, v in trivial:
                v_mean = v.mean()
                k_config = dict(TRIVIAL_CONFIG)[k]
                obj.axhline(y=v_mean.item(), c=k_config[0], linestyle=k_config[1], linewidth=k_config[2], label=k)

            ymin_all, ymax_all = axs[ii].get_ylim()

            # ylim ignore trivial baselines if worse than methods
            if metric in METRICS_HIGHER_BETTER:
                axs[ii].set_ylim((ymin_methods, ymax_all))
            elif metric in METRICS_LOWER_BETTER:
                axs[ii].set_ylim((ymin_all, ymax_methods))
            else:
                axs[ii].set_ylim((ymin_methods, ymax_methods))

        # axis
        axs[ii].set_xticklabels([])
        axs[ii].set_xlabel("")

        # axs[r, i].set_ylabel(metric_format_dict.get(metric, metric))
        axs[ii].set_ylabel("")
        axs[ii].yaxis.grid()
        axs[ii].set_title(metric)
        axs[ii].yaxis.set_major_locator(plt.MaxNLocator(10))

        # legend
        axs[ii].legend([], [], frameon=False)
        handles_, labels_ = axs[ii].get_legend_handles_labels()
        for h_, l_ in zip(handles_, labels_):
            if l_ not in labels:
                handles.append(h_)
                labels.append(l_)

    # multi figure legend
    axs[-1].legend(handles, labels, loc='upper right', bbox_to_anchor=(1.07, -0.15), borderaxespad=1.0)

    # tight layout
    fig.tight_layout()

    plt.savefig(save_path.with_suffix(".pdf"), format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()


def visualize_preds(save_path, g_preds, only=None, ax_width=2.0, ax_height=2.0):

    for trivial in BASELINE_ZERO, BASELINE_RAND, BASELINE_RAND_EDGES:
        if trivial in g_preds:
            del g_preds[trivial]

    methods = list(g_preds.keys())
    methods.remove("__true__")
    ids = sorted(list(g_preds["__true__"].keys()))
    g_preds["true"] = g_preds["__true__"]
    methods = ["true"] + methods
    del g_preds["__true__"]

    if only is not None:
        ids = ids[:10]

    n_rows = len(ids)
    n_cols = len(g_preds)

    sns.set_theme(style="ticks", rc=NEURIPS_RCPARAMS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(ax_width * n_cols, ax_height * n_rows))
    if n_rows == 1:
        axs = [axs]

    for i, (pred_id, axrow) in enumerate(zip(ids, axs)):
        for j, (method, ax) in enumerate(zip(methods, axrow)):
            if i == 0:
                ax.set_title(f'{method}')#, pad=3)

            if pred_id in g_preds[method]:
                ax.matshow(g_preds[method][pred_id], vmin=0, vmax=1, cmap="viridis")
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.axis('off')

    # tight layout
    fig.tight_layout()

    # generate directory if it doesn't exist
    save_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(save_path.with_suffix(".pdf"), format="pdf", facecolor=None,
                dpi=DPI, bbox_inches='tight')

    plt.close()


    # save first prediction of each method
    indiv_path = (save_path.parent / "predictions_individual")
    indiv_path.mkdir(exist_ok=True, parents=True)
    for i, pred_id in enumerate(ids):
        for j, method in enumerate(methods):
            fig, ax = plt.subplots(1, 1, figsize=(ax_width, ax_height))
            ax.matshow(g_preds[method][pred_id], vmin=0, vmax=1, cmap="viridis")
            ax.tick_params(axis='both', which='both', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.axis('off')
            fig.tight_layout()
            plt.savefig((indiv_path / f"{method}_{pred_id}").with_suffix(".pdf"), format="pdf", facecolor=None,
                        dpi=DPI, bbox_inches='tight')

            plt.close()



def visualize_calibration(save_path, domain, g_trues_dict, g_pred_probs_dict, n_bins=10,
                          figsize_calib=(3., 2.8), figsize_freq=(3., 2.8), figsize_calib_all=(3., 3.2)):

    save_path.mkdir(exist_ok=True, parents=True)
    for trivial in BASELINE_ZERO, BASELINE_RAND, BASELINE_RAND_EDGES:
        if trivial in g_pred_probs_dict:
            del g_pred_probs_dict[trivial]

    # make lists
    g_trues = []
    g_probs_methods = defaultdict(list)
    for idx in g_trues_dict.keys():
        g_trues.append(g_trues_dict[idx])
        for method in g_pred_probs_dict.keys():
            g_probs_methods[method].append(g_pred_probs_dict[method][idx])

    if not g_probs_methods:
        return

    # for each method, make calibration plot
    stats_all = defaultdict(dict)

    plot_individual = True
    for method, g_probs in g_probs_methods.items():

        stats = make_calibration_stats(g_trues, g_probs, n_bins=n_bins)
        stats_all[method] = stats

        if plot_individual:

            # plot
            fig, ax = plt.subplots(1, 1, figsize=figsize_calib)

            ############# calibration/reliability plot
            c = CALIBRATION_PLOT_COLOR_SINGLE
            # c = {tup[0]: tup[1] for tup in METHODS_CONFIG_CALIBRATION}[method.split("__")[0]] # catch hparam validation case where names differ
            ax.plot(stats["prob_pred"], stats["prob_true"], color=c)
            ax.scatter(stats["prob_pred"], stats["prob_true"], color=c, s=15, marker="o")

            # diagonal line
            ax.plot(onp.linspace(0, 1, num=100), onp.linspace(0, 1, num=100), color="black", linestyle="dashed")

            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.0, 1.0))
            ax.set_xlabel(CALIBRATION_XLABEL)
            ax.set_ylabel(CALIBRATION_YLABEL)

            ax.xaxis.set_major_locator(plt.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

            plt.grid(visible=True, which='major', axis='both', alpha=0.4)

            fig.tight_layout()

            plt.savefig((save_path / f"{domain}_calibration_{method}").with_suffix(".pdf"),
                        format="pdf", facecolor=None, dpi=DPI, bbox_inches='tight')
            plt.close()

            ############# frequency plot
            # plot
            fig, ax = plt.subplots(1, 1, figsize=figsize_freq)

            # c = {tup[0]: tup[1] for tup in METHODS_CONFIG_CALIBRATION}[method.split("__")[0]] # catch hparam validation case where names differ
            ax.hist(stats["y_pred"], bins=stats["bins"], color=c, density=False)
            ax.set_yscale('log', base=2)

            # make all bars sum to 1 by re-scaling labels
            ylen = len(stats["y_pred"])
            ylabels_raw = [0.001, 0.01, 0.1, 1]
            ylabels_raw_minor = [
                *onp.linspace(0.001, 0.01, 9, endpoint=False),
                *onp.linspace(0.01, 0.1, 9, endpoint=False),
                *onp.linspace(0.1, 1, 9, endpoint=False),
            ]
            ylabel_pos = onp.array(ylabels_raw) * ylen
            ylabel_pos_minor = onp.array(ylabels_raw_minor) * ylen
            ylabel_label = onp.round(ylabel_pos / ylen, 3)
            # ylabel_label_minor = [onp.round(ylabel_pos_minor / ylen, 3)
            ax.set_yticks(ylabel_pos)
            ax.set_yticklabels(ylabel_label)
            ax.set_yticks(ylabel_pos_minor, minor=True)
            ax.set_yticklabels([], minor=True)

            ax.set_xlim((0.0, 1.0))
            ax.set_ylim((0.001 * len(stats["y_pred"]), 1.0 * len(stats["y_pred"])))
            ax.set_xlabel(r"predicted $\hat{p}(g_{i,j}=1\,|\,D)$")
            ax.set_ylabel(r"frequency")

            fig.tight_layout()

            plt.savefig((save_path / f"{domain}_freq_calibration_{method}").with_suffix(".pdf"),
                        format="pdf", facecolor=None, dpi=DPI, bbox_inches='tight')

            plt.close()

    # calibration plots of all methods
    ordering = {method: j for j, (method, *_) in enumerate(METHODS_CONFIG_CALIBRATION)}
    available = sorted(g_probs_methods.keys(), key=lambda method: ordering[method.split("__")[0]])
    is_interv = {tup[0]: tup[2] for tup in METHODS_CONFIG_CALIBRATION}
    for descr, plotted_methods in [
        ("all", ordering),
        ("observ", [method for method in available if not is_interv[method.split("__")[0]]]),
        ("interv", [method for method in available if is_interv[method.split("__")[0]]]),
    ]:
        selected_methods = [method for method in available if method in plotted_methods]
        if not selected_methods:
            continue

        # plot each method
        fig, ax = plt.subplots(1, 1, figsize=figsize_calib_all)
        for method in selected_methods:
            g_probs = g_probs_methods[method]

            stats = make_calibration_stats(g_trues, g_probs, n_bins=n_bins)

            c = {tup[0]: tup[1] for tup in METHODS_CONFIG_CALIBRATION}[method.split("__")[0]] # catch hparam validation case where names differ
            lab = {tup[0]: tup[3] for tup in METHODS_CONFIG_CALIBRATION}[method.split("__")[0]] # catch hparam validation case where names differ
            ax.plot(stats["prob_pred"], stats["prob_true"], color=c, label=lab)
            ax.scatter(stats["prob_pred"], stats["prob_true"], color=c, s=15, marker="o")

        # diagonal line
        ax.plot(onp.linspace(0, 1, num=100), onp.linspace(0, 1, num=100), color="black", linestyle="dashed")

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))
        ax.set_xlabel(CALIBRATION_XLABEL)
        ax.set_ylabel(CALIBRATION_YLABEL)

        ax.xaxis.set_major_locator(plt.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        ax.yaxis.set_major_locator(plt.FixedLocator([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))

        plt.grid(visible=True, which='major', axis='both', alpha=0.4)

        fig.legend(loc="lower left", ncol=2, labelspacing=0.03, bbox_to_anchor=[0.2, 0.0], fontsize='small')

        fig.tight_layout(rect=[0.0, 0.15, 1.0, 1.0])  # default 0, 0, 1, 1
        # fig.tight_layout()

        plt.savefig((save_path / f"{domain}_calibration_{descr}").with_suffix(".pdf"),
                    format="pdf", facecolor=None, dpi=DPI, bbox_inches='tight')

        plt.close()


    # save table of ECE and brier scores
    # create long list of (metric, method, value) tuples which are then aggregated into a df
    df = []
    df_save = []
    for method in g_probs_methods.keys():
        df += [("ece", method, v) for v in stats_all[method]["eces"]]
        # df += [("brier", method, v) for v in stats_all["briers"]]

        df_save += [("prob_pred", method, v, j) for j, v in enumerate(stats_all[method]["prob_pred"])]
        df_save += [("prob_true", method, v, j) for j, v in enumerate(stats_all[method]["prob_true"])]
        df_save += [("y_pred", method, v, j) for j, v in enumerate(stats_all[method]["y_pred"])]
        df_save += [("bins", method, v, j) for j, v in enumerate(stats_all[method]["bins"])]

    df = pd.DataFrame(df, columns=["metric", "method", "val"])

    df_save = pd.DataFrame(df_save, columns=["metric", "method", "val", "index"])
    df_save.to_csv(path_or_buf=(save_path / "df_calibration").with_suffix(".csv"))

    # check whether we need to aggregate at all (not for real data, for which we only measure 1 metric score per method)
    is_singular = all([df.loc[(df.metric == metr) & (df.method == meth)].val.size == 1
        for metr, meth in itertools.product(df.metric.unique(), df.method.unique())])
    if is_singular:
        print("*" * 30 + "\nWarning: singular measurements; ignore std dev, std err or t-tests.")
        df = pd.concat([df, df]).reset_index(drop=True)

    # create table
    table_agg_metrics = [onp.mean, scipy.stats.sem]
    metrname = ["mean", "sem"]
    best_aggmetr = "mean"
    table = df.pivot_table(
        index=["method"],
        columns=["metric"],
        values="val",
        aggfunc=table_agg_metrics,
        dropna=False) \
        .drop(TRIVIAL_METHODS, axis=0, errors="ignore")\
        .swaplevel(-1, -2, axis=1)\
        .sort_index(axis="columns", level="metric")\
        .sort_index()

    # add marker for best
    table = _add_t_test_markers(df, table, best_aggmetr)

    # reorder and rename rows
    method_order = [method for method, *_ in METHODS_CONFIG]
    for method in sorted(table.index):
        if method not in method_order:
            method_order.append(method)
    all_methods_sorted = [method for method in method_order if method in table.index]
    full_table = table.reindex(all_methods_sorted)

    all_metrics = set(m for m, _ in full_table.columns.values)
    all_metrics_ordered = [metric for metric in METRICS_TABLE_ORDERING if metric in all_metrics]
    for metr in all_metrics_ordered:
        full_table = _format_table_str(full_table, metr, metrname)

    full_table = full_table.droplevel(1, axis=1)  # drop the dummy string level

    print(full_table)

    tab_save_path = save_path / f"calibration"
    full_table.to_csv(path_or_buf=tab_save_path.with_suffix(".csv"))
    full_table.to_latex(buf=tab_save_path.with_suffix(".tex"), escape=False)

    return























