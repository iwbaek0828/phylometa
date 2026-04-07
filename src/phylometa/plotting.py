from __future__ import annotations

from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo

MISSING_COLOR = (0.92, 0.92, 0.92, 1.0)


def _is_numeric_series(series: pd.Series) -> bool:
    coerced = pd.to_numeric(series.dropna(), errors="coerce")
    return coerced.notna().all() and len(coerced) > 0


def _categorical_color_map(values: pd.Series) -> Dict[str, tuple]:
    categories = [str(x) for x in pd.Series(values).dropna().astype(str).unique()]
    cmap = plt.get_cmap("tab20")
    return {cat: cmap(i % 20) for i, cat in enumerate(categories)}


def _continuous_color_mapper(values: pd.Series, cmap_name: str = "viridis"):
    numeric = pd.to_numeric(values, errors="coerce")
    vmin = float(np.nanmin(numeric))
    vmax = float(np.nanmax(numeric))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-9
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    return numeric, norm, cmap


def _extract_tip_order(tree) -> List[str]:
    return [tip.name for tip in tree.get_terminals() if tip.name is not None]


def _draw_metadata_strips(ax, aligned: pd.DataFrame, tip_order: List[str], traits: List[str]):
    categorical_legends = {}
    continuous_legends = []
    ax.set_xlim(0, len(traits))
    ax.set_ylim(0, len(tip_order))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(traits)) + 0.5)
    ax.set_xticklabels(traits, rotation=90, fontsize=10)
    ax.set_yticks(np.arange(len(tip_order)) + 0.5)
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    for x, trait in enumerate(traits):
        series = aligned[trait]
        if _is_numeric_series(series):
            numeric, norm, cmap = _continuous_color_mapper(series)
            for y, tip in enumerate(tip_order):
                val = numeric.loc[tip]
                facecolor = MISSING_COLOR if pd.isna(val) else cmap(norm(val))
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=0.6))
            continuous_legends.append((trait, norm, cmap))
        else:
            cat_map = _categorical_color_map(series)
            categorical_legends[trait] = cat_map
            for y, tip in enumerate(tip_order):
                val = series.loc[tip]
                facecolor = MISSING_COLOR if pd.isna(val) else cat_map[str(val)]
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor="white", linewidth=0.6))

    for spine in ax.spines.values():
        spine.set_visible(False)

    return categorical_legends, continuous_legends


def _add_legends(fig, categorical_legends, continuous_legends):
    legend_x = 0.80
    legend_y = 0.95
    for trait, cmap_dict in categorical_legends.items():
        handles = [mpatches.Patch(color=color, label=label) for label, color in cmap_dict.items()]
        if handles:
            fig.legend(handles=handles, title=trait, loc="upper left", bbox_to_anchor=(legend_x, legend_y), frameon=False, fontsize=9, title_fontsize=10)
            legend_y -= min(0.2, 0.05 + 0.03 * len(handles))

    cbar_bottom = 0.06
    cbar_height = 0.16
    cbar_gap = 0.04
    for i, (trait, norm, cmap) in enumerate(continuous_legends):
        ax_cbar = fig.add_axes([0.80, cbar_bottom + i * (cbar_height + cbar_gap), 0.025, cbar_height])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar)
        cbar.set_label(trait, fontsize=9)
        cbar.ax.tick_params(labelsize=8)


def plot_tree_with_traits(
    tree,
    meta: pd.DataFrame,
    id_column: str,
    traits: list[str],
    out_file: str,
    figsize_width: float = 14,
    row_height: float = 0.35,
):
    if id_column not in meta.columns:
        raise ValueError(f"ID column '{id_column}' not found in metadata.")
    for trait in traits:
        if trait not in meta.columns:
            raise ValueError(f"Trait column '{trait}' not found in metadata.")

    tip_order = _extract_tip_order(tree)
    if len(tip_order) == 0:
        raise ValueError("No tip labels found in tree.")

    meta2 = meta.copy()
    meta2[id_column] = meta2[id_column].astype(str)
    if meta2[id_column].duplicated().any():
        dup = meta2.loc[meta2[id_column].duplicated(), id_column].tolist()
        raise ValueError(f"Duplicate IDs in metadata: {dup[:10]}")
    meta2 = meta2.set_index(id_column)

    aligned = pd.DataFrame(index=tip_order)
    for trait in traits:
        aligned[trait] = meta2.reindex(tip_order)[trait]

    n_tips = len(tip_order)
    fig_height = max(6, n_tips * row_height)
    fig = plt.figure(figsize=(figsize_width, fig_height))

    ax_tree = fig.add_axes([0.05, 0.05, 0.45, 0.9])
    Phylo.draw(tree, axes=ax_tree, do_show=False, show_confidence=False)
    ax_tree.set_xlabel("")
    ax_tree.set_ylabel("")

    ax_strip = fig.add_axes([0.53, 0.05, 0.22, 0.9])
    categorical_legends, continuous_legends = _draw_metadata_strips(ax_strip, aligned, tip_order, traits)
    _add_legends(fig, categorical_legends, continuous_legends)

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
