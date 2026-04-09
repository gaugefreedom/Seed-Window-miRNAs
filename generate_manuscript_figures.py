#!/usr/bin/env python3
"""
Generate manuscript figures and tables for the seed-window pilot paper.

Outputs (written to figures/ and papers/paper-02-followup/):
  figures/figure_2_mir124_transitions.png
  figures/figure_3_score_components.png
  figures/figure_3_heatmap.png
  papers/paper-02-followup/manuscript_tables.md
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PAPER = ROOT / "papers" / "paper-02-followup"
PAPER_FIGS = PAPER / "figs"
FIGURES.mkdir(exist_ok=True)
PAPER.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

MAIN_CSV    = DATA / "pilot_run_16.csv"
SUMMARY_CSV = DATA / "pilot_run_16_summary.csv"

# ── colour palette ──────────────────────────────────────────────────────────
COLOUR = {
    "singular":     "#d62728",   # red
    "non_singular": "#1f77b4",   # blue
    "skipped":      "#aec7e8",   # light blue / high-dimensional unprojected
    "timeout":      "#bdbdbd",   # grey
    "regression":   "#d62728",   # same red (published regression singular)
}

COMPONENT_COLOUR = {
    "free_nonfree_flips": "#4c78a8",
    "singular_nonsingular_flips": "#e45756",
    "singularity_type_changes": "#72b7b2",
    "projection_skipped": "#b8b8b8",
    "timeouts": "#6f6f6f",
}

RESOLUTION_ORDER = [
    "benchmark-sliced",
    "published-regression",
    "projection-skipped-due-to-complexity",
    "timeout-preliminary",
]

MIRNA_SHORT = {
    "hsa-miR-124-3p": "miR-124-3p",
    "hsa-miR-29a-3p": "miR-29a-3p",
    "hsa-miR-29c-3p": "miR-29c-3p",
    "hsa-miR-92a-3p": "miR-92a-3p",
    "hsa-miR-486-5p": "miR-486-5p",
    "hsa-let-7i-5p":  "let-7i-5p",
    "hsa-let-7a-5p":  "let-7a-5p",
    "hsa-miR-15a-5p": "miR-15a-5p",
    "hsa-miR-16-5p":  "miR-16-5p",
}


def _window_colour(row: pd.Series) -> str:
    status = row["analysis_status"]
    if status == "timeout-preliminary":
        return COLOUR["timeout"]
    if status == "projection-skipped-due-to-complexity":
        return COLOUR["skipped"]
    # resolved
    if row["is_singular"] is True or str(row["is_singular"]).lower() == "true":
        return COLOUR["singular"]
    return COLOUR["non_singular"]


def _save_figure(fig: plt.Figure, filename: str, dpi: int = 180) -> None:
    for out_dir in (FIGURES, PAPER_FIGS):
        out = out_dir / filename
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out}")


# ── Figure 2: miR-124-3p window transition map ──────────────────────────────

def figure_2_mir124(df: pd.DataFrame) -> None:
    mir = df[df["mirna_id"] == "hsa-miR-124-3p"].copy()
    mir = mir.sort_values(["start", "length"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    bar_height = 0.55
    y_positions = np.arange(len(mir))

    for i, (_, row) in enumerate(mir.iterrows()):
        colour = _window_colour(row)
        ax.barh(
            y=i,
            width=row["length"],
            left=row["start"],
            height=bar_height,
            color=colour,
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        # motif label inside bar
        ax.text(
            row["start"] + row["length"] / 2,
            i,
            row["motif"],
            ha="center", va="center",
            fontsize=8.5,
            fontfamily="monospace",
            color="white" if colour in (COLOUR["singular"], COLOUR["non_singular"]) else "#444444",
            fontweight="bold",
            zorder=4,
        )
        # singularity annotation on right
        if row["analysis_status"] in ("benchmark-sliced", "published-regression"):
            label = "Singular" if str(row["is_singular"]).lower() == "true" else "Non-singular"
            if row["analysis_status"] == "published-regression":
                label += " †"
        elif row["analysis_status"] == "projection-skipped-due-to-complexity":
            label = "Unprojected"
        else:
            label = "Timeout"
        ax.text(
            row["start"] + row["length"] + 0.08,
            i,
            label,
            va="center",
            fontsize=7.5,
            color="#555555",
        )

    # x-axis: nucleotide position
    max_pos = int(mir["start"].max() + mir["length"].max()) + 1
    ax.set_xlim(-0.3, max_pos + 2.0)
    ax.set_xticks(range(max_pos + 1))
    ax.set_xticklabels([str(p) for p in range(max_pos + 1)], fontsize=8)
    ax.set_xlabel("Nucleotide position (0-indexed from mature miRNA)", fontsize=9)

    # y-axis: window label (start_length)
    ylabels = [f"{row['start']},{row['length']}-mer" for _, row in mir.iterrows()]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ylabels, fontsize=8, fontfamily="monospace")
    ax.set_ylabel("Window (start, length)", fontsize=9)

    ax.set_title(
        "Figure 2. Seed-neighborhood window transition map — hsa-miR-124-3p",
        fontsize=11, fontweight="bold", pad=12,
    )

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOUR["singular"],     label="Singular (resolved)"),
        mpatches.Patch(color=COLOUR["non_singular"], label="Non-singular (resolved)"),
        mpatches.Patch(color=COLOUR["skipped"],      label="Unprojected / high-dimensional"),
        mpatches.Patch(color=COLOUR["timeout"],      label="Timeout-preliminary"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
        edgecolor="#cccccc",
    )
    ax.text(
        0.01, -0.14,
        "† Published regression fixture (Lawton et al. framework)",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#666666",
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

    _save_figure(fig, "figure_2_mir124_transitions.png")
    plt.close(fig)


# ── Figure 3: stacked score components ──────────────────────────────────────

def figure_3_score_components(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    status_counts = (
        df.assign(
            projection_skipped=df["analysis_status"].eq("projection-skipped-due-to-complexity"),
            timeouts=df["analysis_status"].eq("timeout-preliminary"),
        )
        .groupby("mirna_id", as_index=False)[["projection_skipped", "timeouts"]]
        .sum()
    )

    plot_df = summary.merge(status_counts, on="mirna_id", how="left").fillna(0)
    plot_df["short_name"] = plot_df["mirna_id"].map(MIRNA_SHORT)
    plot_df = plot_df.sort_values(
        ["provisional_instability_score", "projection_skipped", "timeouts", "short_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    components = [
        ("free_nonfree_flips", "Free/non-free flips", COMPONENT_COLOUR["free_nonfree_flips"], None),
        ("singular_nonsingular_flips", "Singular/non-singular flips", COMPONENT_COLOUR["singular_nonsingular_flips"], None),
        ("singularity_type_changes", "Singularity-type changes", COMPONENT_COLOUR["singularity_type_changes"], None),
        ("projection_skipped", "Projection-skipped windows", COMPONENT_COLOUR["projection_skipped"], "///"),
        ("timeouts", "True timeouts", COMPONENT_COLOUR["timeouts"], "..."),
    ]

    height = max(5.2, 0.55 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")

    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    score_only = (
        plot_df["free_nonfree_flips"]
        + plot_df["singular_nonsingular_flips"]
        + plot_df["singularity_type_changes"]
    ).to_numpy(dtype=float)
    unresolved_only = (
        plot_df["projection_skipped"] + plot_df["timeouts"]
    ).to_numpy(dtype=float)

    for key, label, color, hatch in components:
        values = plot_df[key].to_numpy(dtype=float)
        bars = ax.barh(
            y,
            values,
            left=left,
            height=0.7,
            color=color,
            edgecolor="#ffffff" if hatch is None else "#4f4f4f",
            linewidth=1.0,
            label=label,
        )
        if hatch is not None:
            for bar in bars:
                bar.set_hatch(hatch)
        left += values

    for idx, score in enumerate(score_only):
        if unresolved_only[idx] > 0 and score > 0.5:
            x = score - 0.18
            ha = "right"
            bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.15}
        else:
            x = score + 0.12
            ha = "left"
            bbox = None
        ax.text(
            x,
            idx,
            f"score {int(score)}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#222222",
            bbox=bbox,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["short_name"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count per miRNA", fontsize=10)
    ax.set_title(
        "Figure 3. Decomposition of instability scores and unresolved-window burden",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.95,
        edgecolor="#cccccc",
    )
    ax.text(
        0.0,
        -0.11,
        "Colored segments sum to the provisional instability score. "
        "Gray hatched segments show unresolved windows and are not part of the score.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )

    _save_figure(fig, "figure_3_score_components.png")
    plt.close(fig)


# ── Supplementary figure: instability score heatmap ─────────────────────────

def figure_3_heatmap(summary: pd.DataFrame) -> None:
    df = summary.copy()
    df["short_name"] = df["mirna_id"].map(MIRNA_SHORT)
    df = df.sort_values("provisional_instability_score", ascending=False)

    # build pivot for seaborn heatmap (single column)
    heat_data = df.set_index("short_name")[["provisional_instability_score"]]
    heat_data.columns = ["Instability score"]

    fig, ax = plt.subplots(figsize=(4.5, 5.5))

    # custom colormap: white at 0, saturated blue-green at max
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "instability",
        ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
    )

    score_min = 0
    score_max = float(df["provisional_instability_score"].max())

    sns.heatmap(
        heat_data,
        ax=ax,
        cmap=cmap,
        vmin=score_min,
        vmax=score_max,
        annot=True,
        fmt=".0f",
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=0.5,
        linecolor="#e0e0e0",
        cbar_kws={"label": "Provisional instability score", "shrink": 0.7},
    )

    # mark zero-singular miRNAs with a dot in the annotation
    zero_singular = set(df[df["singular_window_count"] == 0]["short_name"])
    for text_obj in ax.texts:
        label = text_obj.get_text()
        if label == "8":
            y_px = text_obj.get_position()[1]
            row_idx = int(y_px)
            mirna = heat_data.index[row_idx]
            if mirna in zero_singular:
                text_obj.set_text("8 ●")
                text_obj.set_color("#555555")
            else:
                text_obj.set_color("white")
        elif float(label) >= 10:
            text_obj.set_color("white")
        else:
            text_obj.set_color("#333333")

    ax.set_title(
        "Figure 3. Provisional instability scores\nacross the pilot dementia cohort",
        fontsize=10, fontweight="bold", pad=10,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    ax.text(
        0.0, -0.06,
        "● Zero singular windows (free-class transitions only)",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555555",
    )

    _save_figure(fig, "figure_3_heatmap.png")
    plt.close(fig)


# ── Markdown tables ─────────────────────────────────────────────────────────

def _resolution_tier(row: pd.Series) -> str:
    status = row["analysis_status"]
    if status in ("benchmark-sliced", "published-regression"):
        return "Resolved"
    if status == "projection-skipped-due-to-complexity":
        return "Projection-skipped"
    return "Timeout"


def _singular_display(row: pd.Series) -> str:
    tier = _resolution_tier(row)
    if tier == "Timeout":
        return "—"
    if tier == "Projection-skipped":
        return "n/a"
    val = str(row["is_singular"]).lower()
    if val == "true":
        return "Yes"
    if val == "false":
        return "No"
    return "—"


def _singularity_type_display(row: pd.Series) -> str:
    st = str(row["singularity_type"])
    if st in ("nan", "None", "NaN", ""):
        return "—"
    if st == "Timeout":
        return "—"
    if st == "unprojected-high-dimensional":
        return "n/a (unprojected)"
    return st


def table2_md(df: pd.DataFrame, mirna_ids: list) -> str:
    rows = df[df["mirna_id"].isin(mirna_ids)].copy()
    rows = rows.sort_values(["mirna_id", "start", "length"]).reset_index(drop=True)

    lines = [
        "### Table 2. Seed-neighborhood window-level algebraic results",
        "",
        "Windows for `hsa-miR-124-3p` and `hsa-miR-29a-3p`. "
        "Resolution tier: **Resolved** = full Gröbner computation; "
        "**Projection-skipped** = basis computed, elimination bypassed (>6 elimination variables); "
        "**Timeout** = per-window ceiling exceeded. "
        "† Published regression fixture.",
        "",
        "| miRNA | Motif | Len | Free class | Singular | Singularity type | Tier |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in rows.iterrows():
        mirna   = MIRNA_SHORT.get(row["mirna_id"], row["mirna_id"])
        motif   = f"`{row['motif']}`"
        length  = str(int(row["length"]))
        fc      = str(row["free_like_class"]) if str(row["free_like_class"]) not in ("nan", "NaN") else "—"
        sing    = _singular_display(row)
        stype   = _singularity_type_display(row)
        tier    = _resolution_tier(row)
        if row["analysis_status"] == "published-regression":
            tier += " †"
        lines.append(f"| {mirna} | {motif} | {length} | {fc} | {sing} | {stype} | {tier} |")

    return "\n".join(lines)


def table3_md(summary: pd.DataFrame) -> str:
    df = summary.copy()
    df["short_name"] = df["mirna_id"].map(MIRNA_SHORT)
    df = df.sort_values("provisional_instability_score", ascending=False)

    lines = [
        "### Table 3. Per-miRNA instability summary",
        "",
        "Free/non-free flips, singular/non-singular flips, and singularity-type changes "
        "are counted only over adjacent window pairs where both windows have a known "
        "classification in the relevant dimension. "
        "Projection-skipped windows contribute to free-class counts but not singularity counts.",
        "",
        "| miRNA | Windows | Free flips | Sing/NS flips | Type changes | Total transitions | Score |",
        "|---|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        lines.append(
            f"| {row['short_name']} "
            f"| {int(row['n_windows'])} "
            f"| {int(row['free_nonfree_flips'])} "
            f"| {int(row['singular_nonsingular_flips'])} "
            f"| {int(row['singularity_type_changes'])} "
            f"| {int(row['total_transition_count'])} "
            f"| {row['provisional_instability_score']:.1f} |"
        )

    return "\n".join(lines)


def write_tables(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    t2 = table2_md(df, ["hsa-miR-124-3p", "hsa-miR-29a-3p"])
    t3 = table3_md(summary)

    content = "\n\n".join([
        "# Manuscript Tables",
        "",
        t2,
        "",
        t3,
    ])

    out = PAPER / "manuscript_tables.md"
    out.write_text(content)
    print(f"Saved {out}")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    df      = pd.read_csv(MAIN_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    figure_2_mir124(df)
    figure_3_score_components(df, summary)
    figure_3_heatmap(summary)
    write_tables(df, summary)


if __name__ == "__main__":
    main()
