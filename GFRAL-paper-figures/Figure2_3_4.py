#!/usr/bin/env python3
"""
GFRAL Repertoire & Binding Analysis Pipeline

This script loads single-cell and bulk sequence datasets, performs subset
intersections, calculates sequence metrics, performs clustering, and generates
all publication figures.
"""

from copy import deepcopy
from pathlib import Path
import os

from Levenshtein import distance as lev
from matplotlib.ticker import FuncFormatter
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.cluster import dbscan
from venn import venn

# ==========================================
# Global Constants & Paths
# ==========================================
# Paths are resolved relative to this file, so the scripts run from anywhere.
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("STAR_DATA", HERE / "data"))
OUTPUT_DIR = Path(os.environ.get("STAR_FIGURES", HERE / "figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR_SC = DATA_ROOT / "single_cell"
DATA_DIR_BULK = DATA_ROOT / "bulk"
OUTPUT_DIR_IMG = OUTPUT_DIR
OUTPUT_DIR_DATA = OUTPUT_DIR

MICE = ["mouse_1", "mouse_2", "mouse_3", "mouse_4", "mouse_5"]
RELEVANT_DAYS = {
    "mouse_1": ["D17", "D25", "D39", "D46", "D60", "D67"],
    "mouse_2": ["D17", "D25", "D39", "D46", "D60", "D67", "D67_spleen"],
    "mouse_3": ["D3", "D17", "D25", "D39", "D60", "D67"],
    "mouse_4": ["D3", "D17", "D39"],
    "mouse_5": ["D3", "D17", "D39"],
}

COLOR_DICT = {
    # Mouse 1 - Shades of Red
    'Mouse 1 D46 blood': '#FF9999',
    'Mouse 1 D67 Spleen': '#FF6666',
    'Mouse 1 D67 blood': '#CC0000',
    # Mouse 2 - Shades of Blue
    'Mouse 2 D25 blood': '#99CCFF',
    'Mouse 2 D46 blood': '#3399FF',
    'Mouse 2 D67 Spleen': '#0066CC',
    'Mouse 2 D67 blood': '#003366',
    # Mouse 3 - Shades of Green
    'Mouse 3 D46 blood': '#99FF99',
    'Mouse 3 D67 Spleen': '#66CC66',
    'Mouse 3 D67 blood': '#339933',
    # Non present
    'Non present': 'grey',
}


# ==========================================
# Data Loading & Preprocessing
# ==========================================
def load_data():
    """Loads and preprocesses single-cell, bulk, and annotation datasets."""
    # Single-cell datasets
    positive = pd.read_table(DATA_DIR_SC / "Positive.tsv", sep="\t")
    negative = pd.read_table(DATA_DIR_SC / "Negative.tsv", sep="\t")
    total_mice_aligned = pd.read_table(
        DATA_DIR_SC / "total_mice_aligned.tsv", sep="\t"
    )
    all_single_cell = pd.read_table(
        DATA_DIR_SC / "all_single_cell.tsv", sep="\t"
    )
    heavy = pd.read_table(
        DATA_DIR_SC / "all_single_cell_heavy_aligned.tsv", sep="\t"
    )
    paired = pd.read_table(
        DATA_DIR_SC / "all_single_cell_paired.tsv", sep="\t"
    )

    # Bulk datasets
    df_read_gfral = {}
    temp_concat = {}
    for mouse in MICE:
        df_read_gfral[mouse] = {}
        directory_gfral = DATA_DIR_BULK / mouse
        for day in RELEVANT_DAYS[mouse]:
            df_read_gfral[mouse][day] = pd.read_table(
                directory_gfral / day / "df_read.tsv"
            )
        temp_concat[mouse] = pd.concat(
            [df_read_gfral[mouse][day] for day in RELEVANT_DAYS[mouse]],
            ignore_index=True,
        )

    df_read_gfral_all = pd.concat(
        [temp_concat[m] for m in MICE], ignore_index=True
    )

    # Annotations
    df_gfral = pd.read_table(DATA_DIR_SC / "df_GFRAL.tsv", sep="\t")
    df_gfral["new_binding"] = df_gfral["binding"].str.replace("WEAK ", "")

    # SPR Export
    df_table = df_gfral[["aaSeqHeavy", "aaSeqLight", "new_binding"]].copy()
    df_table.rename(columns={"new_binding": "Binding"}, inplace=True)
    df_table.to_excel(OUTPUT_DIR_DATA / "Antibody_SPR.xlsx", index=False)

    # Metadata & Map Annotations
    table_gfral_path = DATA_DIR_SC / "GFRAL_tag.xlsx"
    if table_gfral_path.exists():
        table_gfral = pd.read_excel(table_gfral_path)
        df_gfral["KD"] = df_gfral["Name"].map(
            table_gfral.set_index("GDB Clone Name")["KD (M)"].to_dict()
        )

    seq_iptm_path = DATA_DIR_SC / "sequence_iptm.tsv"
    if seq_iptm_path.exists():
        sequence_iptm = pd.read_table(seq_iptm_path, sep="\t")
        iptm_map = sequence_iptm.set_index("1")["iptm"].to_dict()
        df_gfral["iptm"] = df_gfral["aaSeqHeavy"].map(iptm_map)

    # Specificities
    df_gfral["spe_positive"] = df_gfral["aaSeqHeavy"].isin(positive["aaSeqHeavy"])
    df_gfral["spe_negative"] = df_gfral["aaSeqHeavy"].isin(negative["aaSeqHeavy"])

    def determine_specificity(row):
        if row["spe_positive"] and not row["spe_negative"]:
            return "Only Positive"
        elif row["spe_positive"] and row["spe_negative"]:
            return "Both"
        elif not row["spe_positive"] and row["spe_negative"]:
            return "Only Negative"
        else:
            return "Not present"

    df_gfral["Specificity"] = df_gfral.apply(determine_specificity, axis=1)

    # Drop non-standard index if present
    if 171 in df_gfral.index:
        df_gfral = df_gfral.drop(171)

    # Tissue Processing
    heavy_67 = heavy[heavy["0"].str.contains("D67", regex=False)]
    blood, spleen = {}, {}
    mouse_ids = {"mouse_1": "1152", "mouse_2": "368", "mouse_3": "1149"}

    for mouse, m_id in mouse_ids.items():
        sub = heavy_67[heavy_67["0"].str.contains(m_id)]
        spleen[mouse] = sub[sub["0"].str.contains("Spleen", regex=False)]
        blood[mouse] = sub[sub["0"].str.contains("blood", regex=False)]

    return (
        positive,
        negative,
        total_mice_aligned,
        all_single_cell,
        heavy,
        paired,
        df_read_gfral,
        df_read_gfral_all,
        df_gfral,
        blood,
        spleen,
    )


# ==========================================
# Clustering & Calculations
# ==========================================
def calculate_family_clusters(df_gfral, df_read_gfral_all):
    """Computes Levenshtein DBSCAN clustering and merges neighbor statistics."""

    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])
        return lev(data[i], data[j])

    data = df_gfral["all_seq"].reset_index(drop=True)
    X = np.arange(len(data)).reshape(-1, 1)

    _, labels = dbscan(X, metric=lev_metric, eps=10, min_samples=1)
    df_gfral["family_lev"] = labels

    list_family = df_gfral[df_gfral["iptm"] > 0.7]["family_lev"].unique()
    df_gfral.loc[~df_gfral["family_lev"].isin(list_family), "family_lev"] = 0

    # Map neighbor stats
    nb_iptm = (
        df_read_gfral_all[
            df_read_gfral_all["aaSeqCDR3"].isin(df_gfral["aaSeqCDR3"])
        ]
        .sort_values(by="nb_neighbours_real", ascending=False)
        .drop_duplicates("aaSeqCDR3")
    )
    temp_map = nb_iptm.set_index("aaSeqCDR3")["nb_neighbours_real"].to_dict()
    df_gfral["nb_neighbours"] = df_gfral["aaSeqCDR3"].map(temp_map)

    return df_gfral


def compute_hit_candidates(df_read_gfral):
    """Annotates hit sequences dynamically based on day and mouse thresholds."""
    top_seq = {}
    for mouse in MICE:
        top_seq[mouse] = {}
        for day in RELEVANT_DAYS[mouse]:
            df_day = df_read_gfral[mouse][day]
            if day != "D67":
                top = df_day[df_day["nb_freq"] > 0.000961].reset_index(drop=True)
                cluster_min = 10
            elif mouse == "mouse_1":
                top = df_day[df_day["nb_neighbours_real"] > 90].reset_index(drop=True)
                cluster_min = 10
            elif mouse == "mouse_3":
                top = df_day[df_day["nb_neighbours_real"] > 50].reset_index(drop=True)
                cluster_min = 5
            else:
                top = pd.DataFrame()
                cluster_min = 10

            if len(top) > 0:
                dict_fam = top.value_counts("family").to_dict()
                top["fam_count"] = top["family"].map(dict_fam)
                top = top[top["fam_count"] > cluster_min]
                df_day["Hits_new"] = df_day["aaSeqCDR3"].isin(top["aaSeqCDR3"])
            else:
                df_day["Hits_new"] = False


# ==========================================
# Plotting Functions
# ==========================================
def plot_venn_diagrams(total_mice_aligned, df_gfral, df_read_gfral, blood, spleen, df_read_gfral_all, heavy):
    """Generates all Venn diagram figures."""
    # Figure 1: CDR3 Sets
    cdr3_sets = {
        "Positive": set(
            total_mice_aligned[total_mice_aligned.spe == "Positive"][
                "Heavy aaSeqCDR3"
            ].dropna()
        ),
        "Negative": set(
            total_mice_aligned[total_mice_aligned.spe == "Negative"][
                "Heavy aaSeqCDR3"
            ].dropna()
        ),
        "Binders": set(
            df_gfral[df_gfral.new_binding == "BINDER"]["aaSeqCDR3"].dropna()
        ),
    }
    plt.figure()
    venn(cdr3_sets)
    plt.title("CDR3 Sequences")
    plt.legend("", frameon=False)
    plt.savefig(OUTPUT_DIR_IMG / "SC_pos_neg_binders.pdf")
    plt.close()

    # Figure 2B: bulk vs single cell, per mouse and tissue, at day 67
    panels = [
        ("Mouse 1 Spleen D67", ("mouse_1", "D67"), spleen["mouse_1"]),
        ("Mouse 2 Spleen D67", ("mouse_2", "D67_spleen"), spleen["mouse_2"]),
        ("Mouse 2 Blood D67", ("mouse_2", "D67"), blood["mouse_2"]),
        ("Mouse 3 Spleen D67", ("mouse_3", "D67"), spleen["mouse_3"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (title, (m, day), sc_df) in zip(axes.ravel(), panels):
        bulk_set = set(df_read_gfral[m][day]["aaSeqCDR3"].dropna())
        sc_set = set(sc_df["aaSeqCDR3"].dropna())
        venn2(
            subsets=(bulk_set, sc_set),
            set_labels=("Bulk", "Single cell"),
            ax=ax,
        )
        ax.set_title(title)
        print(
            f"    {title:22s} bulk only {len(bulk_set - sc_set):6d}"
            f"   shared {len(bulk_set & sc_set):5d}"
            f"   single cell only {len(sc_set - bulk_set):5d}"
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR_IMG / "Venn_bulk_vs_sc_D67.pdf")
    plt.close(fig)

    # Figure 3: Bulk Blood vs Spleen
    # "Blood" is filled from D67_spleen and "Spleen" from D67.
    cdr3_blood_spleen = {
        "Blood": set(df_read_gfral["mouse_2"]["D67_spleen"]["aaSeqCDR3"].dropna()),
        "Spleen": set(df_read_gfral["mouse_2"]["D67"]["aaSeqCDR3"].dropna()),
    }
    plt.figure()
    venn2(
        subsets=tuple(cdr3_blood_spleen.values()),
        set_labels=tuple(cdr3_blood_spleen.keys()),
    )
    plt.savefig(OUTPUT_DIR_IMG / "WANN_blood_spleen.pdf")
    plt.close()

    # Figure 4: Bulk vs Single Cell Annotated
    cdr3_b_sc = {
        "Bulk": set(df_read_gfral_all["aaSeqCDR3"].dropna()),
        "Single cell": set(heavy["aaSeqCDR3"].dropna()),
    }
    set1, set2 = cdr3_b_sc.values()
    only1, only2, both = len(set1 - set2), len(set2 - set1), len(set1 & set2)

    plt.figure()
    v = venn2(subsets=(only1, only2, both), set_labels=list(cdr3_b_sc.keys()))
    if v.get_label_by_id("10"):
        v.get_label_by_id("10").set_text(str(only1))
    if v.get_label_by_id("01"):
        v.get_label_by_id("01").set_text(str(only2))
    label11 = v.get_label_by_id("11")
    if label11:
        label11.set_text(str(both))
        x, y = label11.get_position()
        label11.set_position((x, y + 0.08))

    plt.title("CDR3 Sequences")
    plt.savefig(OUTPUT_DIR_IMG / "WANN_b_sc.pdf")
    plt.close()


def plot_scatter_and_histograms(df_gfral):
    """Generates KD, IPTM scatter plots, and specificity histograms."""
    df_gfral["KD"] = df_gfral["KD"].fillna(1e-5)

    def custom_ticks(x, pos):
        return "NB" if abs(x - 1e-5) < 1e-6 else f"{x:.0e}"

    # Cluster Scatter Plot
    colors = [
        '#D3D3D3', '#FF6347', '#1E90FF', '#A52A2A',
        '#FFD700', '#32CD32', '#FF1493'
    ]
    plt.figure()
    sns.scatterplot(
        data=df_gfral, x="KD", y="iptm", hue="family_lev", palette=colors
    )
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    handles, labels = plt.gca().get_legend_handles_labels()
    if labels:
        labels[0] = "Others"
    plt.legend(
        handles, labels, title="Cluster", loc="upper right", ncol=2, frameon=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR_DATA / "scatter_cluster_iptm.pdf")
    plt.close()

    # Neighbors Scatter Plot
    if "nb_neighbours" in df_gfral.columns:
        plt.figure()
        sns.scatterplot(
            data=df_gfral, x="nb_neighbours", y="iptm", palette=colors
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
        plt.savefig(OUTPUT_DIR_DATA / "scatter_neighbours_iptm2.pdf")
        plt.close()

    # KD Specificity Histogram
    plt.figure()
    sns.histplot(
        data=df_gfral,
        x="KD",
        hue="Specificity",
        bins=5,
        multiple="dodge",
        element="step",
        alpha=0.4,
    )
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(FuncFormatter(custom_ticks))
    plt.savefig(OUTPUT_DIR_DATA / "Hist_KD_positive_negative.pdf")
    plt.close()


def plot_sharing_figures(df_gfral):
    """Generates Affinity/IPTM distribution and binding categories plots."""
    df_gfral_new = deepcopy(df_gfral)
    df_gfral_new["KD"] = df_gfral_new["KD"].fillna(1e-6)

    plt.figure(figsize=(8, 5))
    if "colors" in df_gfral_new.columns:
        sns.scatterplot(
            data=df_gfral_new,
            x="KD",
            y="iptm",
            hue="colors",
            palette=list(df_gfral_new["colors"].unique()),
            s=80,
        )
    else:
        sns.scatterplot(data=df_gfral_new, x="KD", y="iptm", s=80)

    plt.xscale("log")
    plt.ylabel("IPTM Alphafold3")
    plt.xlabel("KD")
    plt.xlim(1e-11, 2e-6)

    current_ticks = plt.gca().get_xticks()
    new_labels = [
        'NB' if np.isclose(tick, 1e-6) else f'{tick:.0e}'
        for tick in current_ticks
    ]
    plt.xticks(current_ticks, new_labels)

    plt.savefig(OUTPUT_DIR_IMG / "iptm_KD_sharing_with_NB.pdf")
    plt.close()

    # Binding Categories Plot
    if "Name" in df_gfral.columns:
        df_gfral["cat"] = df_gfral["Name"].str.split("_").str[-1:].str[0]
        dict_binding = {
            "NON BINDER": "NON BINDER",
            "WEAK BINDER": "BINDER",
            "BINDER": "BINDER",
        }
        df_gfral["new_binding"] = df_gfral["binding"].map(dict_binding)
        df_notn = df_gfral[df_gfral["cat"] != "TN"].copy()
        dict_cat = {
            "RHH": "Bulk same cluster",
            "RM": "Bulk same cluster",
            "RH": "Bulk same cluster",
            "RL": "Bulk same cluster",
            "RLL": "Bulk same cluster",
            "VL": "Bulk same cluster",
            "VH": "Bulk same cluster",
            "VM": "Bulk same cluster",
            "O": "STAR output",
            "SC": "Single cell",
        }
        df_notn["new_cat"] = df_notn["cat"].map(dict_cat)

        plt.figure()
        current_palette = sns.color_palette("plasma", 3)
        new_palette = current_palette[2:3] + current_palette[0:1]
        sns.countplot(
            data=df_notn, x="new_cat", hue="new_binding", palette=new_palette
        )
        plt.xlabel("")
        plt.legend(frameon=False)
        plt.savefig(OUTPUT_DIR_IMG / "Binding_categories.pdf")
        plt.close()


def plot_binders_by_sc_category(df_gfral, paired):
    """Fig. 4A: validated binders counted by the single cell sample they appear in.

    A binder is matched to a sample by its full heavy chain against
    all_single_cell_paired.tsv; one binder can appear in several samples.
    Binders matched in no sample form the "Non present" bar.
    """
    binders = df_gfral[df_gfral["new_binding"] == "BINDER"]
    if len(binders) == 0 or "aaSeqHeavy" not in paired.columns:
        return

    mouse_ids = {"1152": "Mouse 1", "368": "Mouse 2", "1149": "Mouse 3"}

    def sample_of(tag):
        tag = str(tag)
        for key, name in mouse_ids.items():
            if key in tag:
                break
        else:
            return None
        day = tag.split("_")[1]
        tissue = "Spleen" if "Spleen" in tag else "blood"
        return f"{name} {day} {tissue}"

    paired = paired.copy()
    paired["sample"] = paired["0"].map(sample_of)
    seq_to_samples = (
        paired.dropna(subset=["sample"])
        .groupby("aaSeqHeavy")["sample"]
        .apply(set)
        .to_dict()
    )

    counts = {k: 0 for k in COLOR_DICT if k != "Non present"}
    n_absent = 0
    for seq in binders["aaSeqHeavy"]:
        samples = seq_to_samples.get(seq, set())
        if not samples:
            n_absent += 1
        for sample in samples:
            if sample in counts:
                counts[sample] += 1
    counts["Non present"] = n_absent

    order = [k for k in COLOR_DICT if counts.get(k, 0) > 0]
    seq_tot = pd.DataFrame(
        {"category": order, "count": [counts[k] for k in order]}
    )

    def short(label):
        if label == "Non present":
            return "Non Present"
        _, _, day, tissue = label.split(" ")
        return f"{day} {'Sp' if tissue == 'Spleen' else 'Bl'}"

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=seq_tot, x="category", y="count",
        order=order, hue="category", hue_order=order,
        palette=COLOR_DICT, legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short(k) for k in order], rotation=45, ha="right")

    # bracket and label the samples belonging to each mouse
    mouse_colors = {"Mouse 1": "#FF6666", "Mouse 2": "#3399FF", "Mouse 3": "#66CC66"}
    top = max(seq_tot["count"]) * 1.06
    for mouse, colour in mouse_colors.items():
        idx = [i for i, k in enumerate(order) if k.startswith(mouse)]
        if not idx:
            continue
        ax.plot([min(idx) - 0.4, max(idx) + 0.4], [top, top],
                color=colour, linewidth=2, clip_on=False)
        ax.text((min(idx) + max(idx)) / 2, top * 1.02, mouse,
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, top * 1.12)
    ax.set_title("Binders by single cell categories", pad=24)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR_IMG / "Binders_by_sc_category.pdf")
    plt.close(fig)

    print(f"    {len(binders)} binders, {n_absent} absent from the single cell data")
    for k in order:
        print(f"    {k:22s} {counts[k]:4d}")


def plot_binder_sharing(df_gfral, paired):
    """Fig. 4E: abundance profiles of the binders shared between mice.

    A binder is "shared" when its heavy chain turns up in samples from more than
    one mouse. Each group of bars is one such binder, labelled by its row number
    in the binder table, and the bars are its cell counts in each sample.

    The list is deduplicated by CDR3, keeping the most widely shared row of
    each group, so each sequence appears once.
    """
    binders = df_gfral[df_gfral["new_binding"] == "BINDER"].reset_index(drop=True)
    if len(binders) == 0 or "aaSeqHeavy" not in paired.columns:
        return

    mouse_ids = {"1152": "Mouse 1", "368": "Mouse 2", "1149": "Mouse 3"}

    def sample_of(tag):
        tag = str(tag)
        for key, name in mouse_ids.items():
            if key in tag:
                break
        else:
            return None
        day = tag.split("_")[1]
        tissue = "Spleen" if "Spleen" in tag else "blood"
        return f"{name} {day} {tissue}"

    paired = paired.copy()
    paired["sample"] = paired["0"].map(sample_of)
    paired = paired.dropna(subset=["sample"])
    per_sample = (
        paired.groupby(["aaSeqHeavy", "sample"]).size().rename("count").reset_index()
    )
    mice_of = (
        paired.assign(mouse=paired["sample"].str.slice(0, 7))
        .groupby("aaSeqHeavy")["mouse"].apply(set).to_dict()
    )

    shared = binders[
        binders["aaSeqHeavy"].map(lambda q: len(mice_of.get(q, set())) > 1)
    ].copy()
    n_rows = len(shared)
    # one bar group per CDR3; where several rows share a CDR3, keep the one
    # whose heavy chain is seen in the most mice
    shared["_nmice"] = shared["aaSeqHeavy"].map(lambda q: len(mice_of.get(q, set())))
    shared = (
        shared.sort_values("_nmice", ascending=False)
        .drop_duplicates("aaSeqCDR3")
        .sort_index()
    )

    rows = []
    for idx, seq in shared["aaSeqHeavy"].items():
        for _, r in per_sample[per_sample["aaSeqHeavy"] == seq].iterrows():
            rows.append({"seq": str(idx), "sample": r["sample"], "count": r["count"]})
    if not rows:
        return
    new_seq_tot = pd.DataFrame(rows)

    seq_order = sorted(set(new_seq_tot["seq"]), key=int)
    sample_order = [k for k in COLOR_DICT if k != "Non present"]

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(
        data=new_seq_tot, x="seq", y="count", hue="sample",
        order=seq_order, hue_order=sample_order,
        palette=COLOR_DICT, errorbar=None,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Sequence")
    ax.set_ylabel("count")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    plt.title("Sharing of binders")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR_IMG / "Binders_sharing.pdf")
    plt.close()

    n3 = sum(1 for q in shared["aaSeqHeavy"]
             if len(mice_of.get(q, set())) == 3)
    print(f"    {n_rows} binder rows shared across mice -> {len(seq_order)} "
          f"distinct sequences ({n3} in all three mice, {len(seq_order) - n3} in two): "
          f"{', '.join(seq_order)}")


def plot_hits_per_day(df_read_gfral):
    """Fig. 3B: number of STAR hits per mouse and day, for the three responders.

    Uses the `hits` column stored in df_read.tsv.
    """
    responders = ["mouse_1", "mouse_2", "mouse_3"]
    rows = []
    for mouse in responders:
        for day in RELEVANT_DAYS[mouse]:
            df_day = df_read_gfral[mouse][day]
            if len(df_day) == 0 or "hits" not in df_day.columns:
                continue
            n = int(df_day["hits"].sum())
            if n:
                rows.append({"mouse": mouse, "day": day, "count": n})
    if not rows:
        return
    hits = pd.DataFrame(rows)

    day_order = [d for d in ["D3", "D17", "D25", "D39", "D46", "D60", "D67",
                             "D67_spleen"] if d in set(hits["day"])]
    palette = {"mouse_1": "#FF7F7F", "mouse_2": "#3399FF", "mouse_3": "#2CA02C"}

    plt.figure()
    ax = sns.barplot(
        data=hits, x="day", y="count", hue="mouse",
        order=day_order, hue_order=responders, palette=palette,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("count")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Mouse 1", "Mouse 2", "Mouse 3"], frameon=False)
    plt.title("Number of hits")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR_IMG / "Hits_per_mouse_day.pdf")
    plt.close()

    for _, r in hits.sort_values(["day", "mouse"]).iterrows():
        print(f"    {r['mouse']}  {r['day']:11s} {r['count']:5d} hits")


def plot_neighbor_curves(df_read_gfral, df_read_gfral_all):
    """Plots neighbor accumulation curves across days."""
    df_39 = df_read_gfral_all[df_read_gfral_all["day"] == "D39"]
    df_60 = df_read_gfral_all[df_read_gfral_all["day"] == "D60"]

    if len(df_39) == 0 or len(df_60) == 0:
        return

    # Cumulative calculations for Day 39
    max_39 = int(df_39["nb_neighbours_real"].max()) + 1
    mean_neigh = pd.DataFrame(index=np.arange(max_39))
    for m in MICE:
        mean_neigh[m] = df_read_gfral[m]["D39"][
            "nb_neighbours_real"
        ].value_counts()
    mean_neigh.fillna(0, inplace=True)
    cum_39 = mean_neigh.sum() - mean_neigh.cumsum()
    cum_39["mean"] = cum_39.mean(axis=1)

    # Cumulative calculations for Day 60
    max_60 = int(df_60["nb_neighbours_real"].max()) + 1
    mean_neigh60 = pd.DataFrame(index=np.arange(max_60))
    for m in MICE[0:3]:
        mean_neigh60[m] = df_read_gfral[m]["D60"][
            "nb_neighbours_real"
        ].value_counts()
    mean_neigh60.fillna(0, inplace=True)
    cum_60 = mean_neigh60.sum() - mean_neigh60.cumsum()
    cum_60["mean"] = cum_60.mean(axis=1)

    plt.figure()
    plt.plot(cum_39["mean"], linewidth=2.3, c="dodgerblue", label="Day 39")
    plt.plot(cum_60["mean"], linewidth=2.3, c="mediumorchid", label="Day 60")
    plt.plot([300] * 4, [0, 100, 1000, 90000], "--", c="darkmagenta", label="Threshold")

    for m in MICE:
        plt.plot(cum_39[m], alpha=0.4, c="dodgerblue")
    for m in MICE[0:3]:
        plt.plot(cum_60[m], alpha=0.4, c="mediumorchid")

    plt.ylabel("Counts")
    plt.xlabel("Neighbours")
    plt.yscale("log")
    plt.legend(frameon=False)
    plt.savefig(OUTPUT_DIR_IMG / "Neighbours_nolog.pdf")
    plt.close()


# ==========================================
# Main Execution Workflow
# ==========================================
def main():
    print("Loading and preparing datasets...")
    (
        positive,
        negative,
        total_mice_aligned,
        all_single_cell,
        heavy,
        paired,
        df_read_gfral,
        df_read_gfral_all,
        df_gfral,
        blood,
        spleen,
    ) = load_data()

    print("Executing family clustering and neighbor annotation...")
    df_gfral = calculate_family_clusters(df_gfral, df_read_gfral_all)
    compute_hit_candidates(df_read_gfral)

    # Re-concatenate all datasets after hit tagging
    temp_concat = [
        pd.concat(
            [df_read_gfral[m][d] for d in RELEVANT_DAYS[m]],
            ignore_index=True,
        )
        for m in MICE
    ]
    df_read_gfral_all = pd.concat(temp_concat, ignore_index=True)

    print("Generating figures...")
    plot_venn_diagrams(
        total_mice_aligned,
        df_gfral,
        df_read_gfral,
        blood,
        spleen,
        df_read_gfral_all,
        heavy,
    )
    plot_scatter_and_histograms(df_gfral)
    plot_sharing_figures(df_gfral)
    plot_binders_by_sc_category(df_gfral, paired)
    plot_binder_sharing(df_gfral, paired)
    plot_hits_per_day(df_read_gfral)
    plot_neighbor_curves(df_read_gfral, df_read_gfral_all)

    print("Exporting updated dataset...")
    df_gfral.to_csv(
        OUTPUT_DIR_DATA / "df_GFRAL_KD_iptm.tsv", sep="\t", index=False
    )

    print("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()