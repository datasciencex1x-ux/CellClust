"""
cellclust.plots
===============
Manuscript-grade matplotlib helpers (600 dpi, RGB, white background).

Each function returns a matplotlib ``Figure`` for further customization;
saving is delegated to ``save_figure`` so that the output is always
flattened to plain RGB on a white background, which is required for
correct embedding in Microsoft Word.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy.stats import gaussian_kde

from .config import (
    COLORS, ALG_COLORS, DPI_FINAL, MUSCLES,
    ALGORITHMS, TREATNMD_THRESHOLD_UM,
)


def configure_rcparams():
    """Set matplotlib defaults for manuscript-grade figures."""
    plt.rcParams.update({
        "font.family":         "DejaVu Sans",
        "font.size":           10,
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "savefig.facecolor":   "white",
        "savefig.edgecolor":   "none",
        "savefig.transparent": False,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.30,
        "savefig.dpi":         DPI_FINAL,
        "axes.edgecolor":      "#333333",
        "axes.labelcolor":     "#1A1918",
        "xtick.color":         "#1A1918",
        "ytick.color":         "#1A1918",
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      0.8,
        "xtick.labelsize":     9,
        "ytick.labelsize":     9,
        "axes.labelsize":      10,
        "legend.frameon":      False,
        "legend.fontsize":     9,
    })


def _panel_label(ax, letter, x=-0.10, y=1.04, size=12):
    ax.text(x, y, f"({letter})", transform=ax.transAxes,
             fontsize=size, fontweight="bold", color=COLORS["ink"],
             ha="left", va="bottom")


def _force_white_background(fig):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for ax in fig.get_axes():
        ax.set_facecolor("white")
        ax.patch.set_alpha(1.0)


def save_figure(fig, path, dpi=DPI_FINAL):
    """Save figure as PNG and flatten to plain RGB on white background.

    Word renders RGBA PNGs as gray boxes on some systems; flattening
    removes the alpha channel and guarantees correct embedding.
    """
    _force_white_background(fig)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    # Flatten RGBA -> RGB on white background (only if file exists)
    try:
        img = Image.open(path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        bg.save(path, "PNG", optimize=True)
    except FileNotFoundError:
        pass  # savefig failed silently; skip flatten step


# ---------------------------------------------------------------
# Figure 1 - Dataset overview (KDE pooled + per-animal violins)
# ---------------------------------------------------------------
def plot_dataset_overview(data_by_muscle):
    configure_rcparams()
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        gridspec_kw={"hspace": 0.50, "wspace": 0.30,
                      "left": 0.08, "right": 0.97,
                      "top": 0.94, "bottom": 0.08},
    )
    for i, m in enumerate(MUSCLES):
        ax = axes[0, i]
        feret = data_by_muscle[m]["feret"].values
        color = COLORS["sol"] if m == "SOL" else COLORS["gas"]
        xs = np.linspace(feret.min(), feret.max(), 400)
        kde = gaussian_kde(feret)
        ys = kde(xs)
        ax.fill_between(xs, ys, color=color, alpha=0.22)
        ax.plot(xs, ys, color=color, lw=2.0)
        ax.axvline(feret.mean(), color=COLORS["ink"], lw=0.8, ls="--", alpha=0.6)
        ax.axvline(TREATNMD_THRESHOLD_UM, color=color, lw=0.8, ls=":", alpha=0.6)
        ax.text(0.97, 0.95,
            f"n = {len(feret):,}\nmu = {feret.mean():.2f}\nsigma = {feret.std():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=0.8))
        ax.set_xlabel("Min Feret diameter (um)")
        ax.set_ylabel("Density")
        ax.set_title(m, fontsize=11, fontweight="bold", color=color, pad=8)
        _panel_label(ax, "a" if i == 0 else "b")

    for i, m in enumerate(MUSCLES):
        ax = axes[1, i]
        color = COLORS["sol"] if m == "SOL" else COLORS["gas"]
        sub = data_by_muscle[m]
        animals = sub.groupby("animal")["feret"].mean().sort_values().index.tolist()
        for j, a in enumerate(animals):
            vals = sub.loc[sub["animal"] == a, "feret"].values
            if len(vals) < 5:
                continue
            kde2 = gaussian_kde(vals)
            ys2 = np.linspace(vals.min(), vals.max(), 200)
            xs2 = kde2(ys2)
            xs2 = xs2 / xs2.max() * 0.40
            ax.fill_betweenx(ys2, j - xs2, j + xs2, color=color, alpha=0.55, lw=0)
            ax.scatter([j], [np.median(vals)], color=COLORS["ink"], s=12,
                        zorder=5, edgecolor="white", lw=0.6)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(animals)} animals (sorted by mean diameter)")
        ax.set_ylabel("Min Feret diameter (um)")
        _panel_label(ax, "c" if i == 0 else "d")
    return fig


# ---------------------------------------------------------------
# Figure 2 - K-selection criteria (4 criteria x 2 muscles)
# ---------------------------------------------------------------
def plot_k_selection(K_criteria):
    configure_rcparams()
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 7.5),
        gridspec_kw={"hspace": 0.60, "wspace": 0.50,
                      "left": 0.06, "right": 0.98,
                      "top": 0.93, "bottom": 0.10},
    )
    criteria = [
        ("inertia", "elbow_k_opt", "Inertia (K-Means)"),
        ("bic",     "bic_k_opt",   "BIC (GMM)"),
        ("aic",     "aic_k_opt",   "AIC (GMM)"),
        ("gap",     "gap_k_opt",   "Gap statistic"),
    ]
    for row, m in enumerate(MUSCLES):
        color = COLORS["sol"] if m == "SOL" else COLORS["gas"]
        c = K_criteria[m]
        K = c["K"]
        for col, (key, opt_key, ylabel) in enumerate(criteria):
            ax = axes[row, col]
            if key == "gap":
                y = c["gap"]; se = c["gap_se"]
                ax.errorbar(K, y, yerr=se, fmt="-o", color=color,
                             ecolor=COLORS["muted"], capsize=2, lw=1.5,
                             markersize=5, mec="white", mew=0.6)
            else:
                y = c[key]
                ax.plot(K, y, "-o", color=color, lw=1.5, markersize=5,
                         mec="white", mew=0.6)
            k_opt = c[opt_key]
            idx = K.index(k_opt)
            ax.scatter([k_opt], [y[idx]], s=140, facecolors="none",
                        edgecolors=COLORS["sol"], lw=2.0, zorder=10)
            ax.annotate(f"k = {k_opt}", xy=(k_opt, y[idx]),
                         xytext=(10, 10), textcoords="offset points",
                         color=COLORS["sol"], fontsize=9, fontweight="bold")
            ax.set_xticks(K)
            ax.set_xlabel("k")
            ax.set_ylabel(f"{ylabel}\n({m})" if col == 0 else ylabel)
            _panel_label(ax, chr(97 + row * 4 + col),
                          x=-0.32 if col == 0 else -0.18)
    return fig


# ---------------------------------------------------------------
# Figure 3 - Voting matrix
# ---------------------------------------------------------------
def plot_voting_matrix(bench_df):
    configure_rcparams()
    fig, ax = plt.subplots(
        figsize=(11, 7),
        gridspec_kw={"left": 0.20, "right": 0.93,
                      "top": 0.93, "bottom": 0.10},
    )
    data_rows, row_labels = [], []
    ks = list(range(2, 9))
    for m in MUSCLES:
        for alg in ALGORITHMS:
            d = bench_df[(bench_df["Muscle"] == m) &
                         (bench_df["Algorithm"] == alg)].sort_values("k")
            sil_vals = d["Silhouette"].values
            sil_n = sil_vals / sil_vals.max()
            data_rows.append(sil_n)
            row_labels.append(f"{m}  -  {alg}")
    matrix = np.array(data_rows)
    cmap = LinearSegmentedColormap.from_list(
        "vote", ["white", "#F4ECD8", "#7A9B7E", "#C9A961", "#C66B4A"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i, row in enumerate(data_rows):
        argmax = int(np.argmax(row))
        ax.text(argmax, i, f"k={ks[argmax]}", ha="center", va="center",
                 color=COLORS["ink"], fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                            ec=COLORS["sol"], lw=1.2))
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Number of clusters (k)")
    ax.axhline(len(ALGORITHMS) - 0.5, color=COLORS["ink"], lw=1.0, alpha=0.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Silhouette (normalized per row)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return fig


# ---------------------------------------------------------------
# Figure 4 - Composite benchmark
# ---------------------------------------------------------------
def plot_composite_benchmark(bench_by_muscle):
    configure_rcparams()
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"wspace": 0.55, "left": 0.15, "right": 0.97,
                      "top": 0.92, "bottom": 0.13},
    )
    for i, m in enumerate(MUSCLES):
        ax = axes[i]
        bench_m = bench_by_muscle[m].sort_values(
            "Composite", ascending=False).reset_index(drop=True)
        algs = bench_m["Algorithm"].values
        composites = bench_m["Composite"].values
        bar_colors = [ALG_COLORS[a] for a in algs]
        y_pos = np.arange(len(algs))[::-1]
        ax.barh(y_pos, composites, color=bar_colors, edgecolor="white",
                 lw=0.8, height=0.72, alpha=0.92)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(algs, fontsize=9.5)
        for j, c in enumerate(composites):
            ax.text(c + 0.012, y_pos[j], f"{c:.3f}", va="center",
                     fontsize=8.5, color=COLORS["ink"], family="monospace")
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Composite score  (Sil + CH + 1/DB normalized)")
        ax.set_title(m, fontsize=12, fontweight="bold",
                      color=COLORS["sol"] if m == "SOL" else COLORS["gas"],
                      pad=8)
        ax.grid(axis="x", alpha=0.20, ls=":", color=COLORS["muted"])
        _panel_label(ax, "a" if i == 0 else "b", x=-0.36)
    return fig


# ---------------------------------------------------------------
# Figure 5 - ARI matrices 8x8
# ---------------------------------------------------------------
def plot_ari_matrices(mat_sol, mat_gas):
    configure_rcparams()
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"wspace": 0.45, "left": 0.10, "right": 0.95,
                      "top": 0.92, "bottom": 0.22},
    )
    cmap = LinearSegmentedColormap.from_list(
        "ari", ["white", "#F4ECD8", "#7A9B7E", "#C9A961", "#C66B4A"]
    )
    for i, (m, mat) in enumerate(zip(MUSCLES, [mat_sol, mat_gas])):
        ax = axes[i]
        im = ax.imshow(mat.values, cmap=cmap, vmin=0, vmax=1)
        for r in range(len(mat)):
            for c in range(len(mat)):
                v = mat.values[r, c]
                tc = "white" if v > 0.55 else COLORS["ink"]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                         color=tc, fontsize=8.5, fontweight="bold",
                         family="monospace")
        ax.set_xticks(range(len(mat)))
        ax.set_yticks(range(len(mat)))
        ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(mat.index, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(m, fontsize=12, fontweight="bold",
                      color=COLORS["sol"] if m == "SOL" else COLORS["gas"],
                      pad=8)
        cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label("Adjusted Rand Index", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        _panel_label(ax, "a" if i == 0 else "b", x=-0.22, y=1.04)
    return fig


# ---------------------------------------------------------------
# Figure 6 - Cluster signatures (stacked histograms)
# ---------------------------------------------------------------
def plot_cluster_signatures(data_by_muscle, labels_by_muscle,
                              cluster_names=("Low", "High")):
    configure_rcparams()
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8.5),
        gridspec_kw={"hspace": 0.45, "left": 0.08, "right": 0.97,
                      "top": 0.95, "bottom": 0.08},
    )
    for i, m in enumerate(MUSCLES):
        ax = axes[i]
        color = COLORS["sol"] if m == "SOL" else COLORS["gas"]
        feret = data_by_muscle[m]["feret"].values
        labels = labels_by_muscle[m]
        bins = np.linspace(feret.min(), feret.max(), 70)
        bottoms = np.zeros(len(bins) - 1)
        cluster_colors = [COLORS["low"], COLORS["high"]]
        for cl in range(2):
            counts, _ = np.histogram(feret[labels == cl], bins=bins)
            ax.bar(bins[:-1], counts, width=np.diff(bins),
                    bottom=bottoms, color=cluster_colors[cl], alpha=0.92,
                    ec="white", lw=0.3,
                    label=f"{cluster_names[cl]} "
                          f"(mu={feret[labels==cl].mean():.1f} um, "
                          f"n={(labels==cl).sum():,}, "
                          f"{100*(labels==cl).mean():.1f}%)")
            bottoms += counts
        ymax = bottoms.max()
        for cl in range(2):
            mu = feret[labels == cl].mean()
            ax.axvline(mu, color=cluster_colors[cl], lw=1.3,
                        ls=(0, (3, 1.5)), alpha=0.85)
        ax.axvline(TREATNMD_THRESHOLD_UM, color=COLORS["ink"],
                    lw=0.7, ls=":", alpha=0.5)
        pct_atro = 100 * (feret < TREATNMD_THRESHOLD_UM).mean()
        ax.text(TREATNMD_THRESHOLD_UM, ymax * 0.96,
                 f"  {TREATNMD_THRESHOLD_UM:.0f} um (Treat-NMD)\n"
                 f"  {pct_atro:.2f}% below",
                 color=COLORS["ink"], fontsize=8, va="top",
                 style="italic", alpha=0.7)
        ax.set_xlabel("Min Feret diameter (um)")
        ax.set_ylabel("Number of fibers")
        ax.set_title(m, fontsize=12, fontweight="bold", color=color, pad=8)
        ax.set_ylim(0, ymax * 1.15)
        ax.legend(fontsize=9, loc="upper right", frameon=False)
        _panel_label(ax, "a" if i == 0 else "b", x=-0.07, y=1.05)
    return fig


# ---------------------------------------------------------------
# Figure 7 - Diet x Week (per-animal boxplots)
# ---------------------------------------------------------------
def plot_diet_x_week(per_animal, diet_eff,
                       cluster_names=("Low", "High")):
    configure_rcparams()
    fig = plt.figure(figsize=(13, 9.5))
    gs = GridSpec(3, 2, figure=fig,
                    height_ratios=[0.05, 1, 1],
                    hspace=0.55, wspace=0.30,
                    left=0.10, right=0.97, top=0.96, bottom=0.07)
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["ncd"], alpha=0.5,
                       edgecolor=COLORS["ncd"], lw=1.5, label="NCD"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["hfd"], alpha=0.5,
                       edgecolor=COLORS["hfd"], lw=1.5, label="HFD"),
    ]
    leg_ax.legend(handles=handles, loc="center", ncol=2,
                   frameon=False, fontsize=12, handletextpad=0.6,
                   columnspacing=3.0)
    weeks = [8, 12, 16]
    for row_idx, m in enumerate(MUSCLES):
        sub = per_animal[per_animal["Musculo"] == m]
        color_m = COLORS["sol"] if m == "SOL" else COLORS["gas"]
        for col_idx, cl in enumerate(cluster_names):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            for wi, week in enumerate(weeks):
                for di, diet in enumerate(["NCD", "HFD"]):
                    pos = wi + (di - 0.5) * 0.36
                    vals = sub.loc[(sub["diet"] == diet) &
                                    (sub["week"] == week), f"pct_{cl}"].values
                    if len(vals) == 0:
                        continue
                    d_color = COLORS["ncd"] if diet == "NCD" else COLORS["hfd"]
                    ax.boxplot(vals, positions=[pos], widths=0.30,
                                patch_artist=True, showfliers=False, zorder=2,
                                boxprops=dict(facecolor=d_color, alpha=0.45,
                                              edgecolor=d_color, lw=1.0),
                                whiskerprops=dict(color=d_color, lw=1.0),
                                capprops=dict(color=d_color, lw=1.0),
                                medianprops=dict(color=COLORS["ink"], lw=1.2))
                    rng_l = np.random.default_rng(42 + wi + di)
                    jitter = rng_l.uniform(-0.06, 0.06, size=len(vals))
                    ax.scatter(np.full_like(vals, pos) + jitter, vals,
                                color=d_color, s=28, alpha=0.85,
                                edgecolor="white", lw=0.7, zorder=3)
            ax.set_xticks(range(len(weeks)))
            ax.set_xticklabels([f"Week {w}" for w in weeks])
            ax.set_xlim(-0.5, len(weeks) - 0.5)
            ax.set_xlabel("Time point")
            ax.set_ylabel(f"% {cl} fibers")
            ax.grid(axis="y", alpha=0.18, ls=":")
            if col_idx == 0:
                ax.set_title(m, fontsize=12, fontweight="bold",
                              color=color_m, loc="left", pad=10)
            y_data = sub[f"pct_{cl}"].values
            ymin = y_data.min(); ymax = y_data.max()
            yspan = ymax - ymin
            ax.set_ylim(max(0, ymin - 0.05 * yspan), ymax + 0.22 * yspan)
            de_row = diet_eff[(diet_eff["Muscle"] == m) &
                                (diet_eff["Cluster"] == cl)]
            if len(de_row):
                de = de_row.iloc[0]
                ax.text(0.02, 0.96,
                         f"NCD vs HFD: p = {de['Welch_p']:.3f}    "
                         f"Cohen's d = {abs(de['Cohen_d']):.2f}",
                         transform=ax.transAxes, ha="left", va="top",
                         fontsize=8.5, color=COLORS["ink"], family="monospace",
                         bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                    ec="none", alpha=1.0))
            _panel_label(ax, "abcd"[row_idx * 2 + col_idx])
    return fig
