#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:11:42 2026


@author: awalczak
"""


import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths are resolved relative to this file, so the scripts run from anywhere.
HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("STAR_DATA", HERE / "data"))
OUTPUT_DIR = Path(os.environ.get("STAR_FIGURES", HERE / "figures"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------
# Read spreadsheet
# -----------------------


df = pd.read_excel(DATA_ROOT / "spr" / "SI_Dataset_1_withnumbers.xlsx")


# Column names
category_col = "category"          # Column B
binding_col = "BINDING STATUS"     # Column D


categories = ["bulk", "STAR", "SC"]


nonbinder = []
binder = []


for c in categories:
    nonbinder.append(
        ((df[category_col] == c) &
         (df[binding_col] == "NON BINDER")).sum()
    )


    binder.append(
        ((df[category_col] == c) &
         (df[binding_col].isin(["BINDER", "WEAK BINDER"]))).sum()
    )


# -----------------------
# Print counts
# -----------------------


print("counts")
for c, nb, b in zip(categories, nonbinder, binder):
    print(f"{c:5s}: NON BINDER = {nb}, BINDER + WEAK BINDER = {b}")


# -----------------------
# Plot
# -----------------------


x = np.arange(len(categories))
width = 0.38


fig, ax = plt.subplots(figsize=(4.2, 4.0))


orange = "#E39A50"
purple = "#6A1B9A"


ax.bar(
    x - width/2,
    nonbinder,
    width=width,
    color=orange,
    edgecolor=orange,
    label="Non binder"
)


ax.bar(
    x + width/2,
    binder,
    width=width,
    color=purple,
    edgecolor=purple,
    label="Binder"
)


ax.set_xticks(x)
ax.set_xticklabels(
    ["Bulk same cluster", "STAR", "Single cell"],
    fontsize=12
)


ax.set_ylabel("count", fontsize=10)
ax.tick_params(axis="both", labelsize=10)


ax.legend(frameon=False, fontsize=10)


# -----------------------
# Save figure
# -----------------------


fig.tight_layout()


fig.savefig(
    OUTPUT_DIR / "Fig3C_new.pdf",
    format="pdf",
    bbox_inches="tight"
)


fig.savefig(
    OUTPUT_DIR / "Fig3C_new.png",
    dpi=600,
    bbox_inches="tight"
)


# -----------------------
# Show figure
# -----------------------


plt.show()
