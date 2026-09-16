#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:38:18 2026


@author: awalczak
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# File
INPUT_FILE = "dfGFRALKDiptm2.xls"


# Read Excel file
df = pd.read_excel(INPUT_FILE)


# Make sure IPTM is numeric
df["iptm"] = pd.to_numeric(df["iptm"], errors="coerce")


# Create the two vectors
binder = df.loc[df["binding"] == "BINDER", "iptm"].dropna().values
non_binder = df.loc[df["binding"] == "NON BINDER", "iptm"].dropna().values


# Check: print the means
print(f"Mean iptm (BINDER):     {np.mean(binder):.4f}")
print(f"Mean iptm (NON BINDER): {np.mean(non_binder):.4f}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# File
INPUT_FILE = "dfGFRALKDiptm2.xls"


# Read Excel file
df = pd.read_excel(INPUT_FILE)


# Make sure IPTM is numeric
df["iptm"] = pd.to_numeric(df["iptm"], errors="coerce")


# Create the two vectors
binder = df.loc[df["binding"] == "BINDER", "iptm"].dropna().values
non_binder = df.loc[df["binding"] == "NON BINDER", "iptm"].dropna().values


# Check: print the means
print(f"Mean iptm (BINDER):     {np.mean(binder):.4f}")
print(f"Mean iptm (NON BINDER): {np.mean(non_binder):.4f}")


# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(6, 5))


positions = [1, 1.6]  # reduced gap between columns


# Scatter only (jittered so points do not overlap)
np.random.seed(42)  # reproducible jitter
for i, (vec, label, color) in enumerate(zip(
        [binder, non_binder],
        ["Binder", "Non binder"],
        ["purple", "orange"])):
    x = np.random.normal(positions[i], 0.1, size=len(vec))
    ax.scatter(x, vec, alpha=0.7, s=30, color=color, edgecolor=color, label=label)


# Calculate means and standard errors
means = [np.mean(binder), np.mean(non_binder)]
stds  = [np.std(binder, ddof=1), np.std(non_binder, ddof=1)]
ns    = [len(binder), len(non_binder)]
sems  = [s / np.sqrt(n) for s, n in zip(stds, ns)]


# Add mean lines and error bars
for i, (mean_val, sem, color) in enumerate(zip(means, sems, ["purple", "orange"])):
    ax.hlines(mean_val, positions[i] - 0.2, positions[i] + 0.2,
              color=color, linewidth=2)
    ax.errorbar(positions[i], mean_val, yerr=sem,
                color=color, capsize=5, capthick=2, elinewidth=2)


# Axis styling (fontsize 10pt, no x-label)
ax.set_xticks(positions)
ax.set_xticklabels(["Binder", "Non Binder"])
ax.set_ylabel('IPTM Alphafold3', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
#ax.legend(fontsize=8)


plt.tight_layout()


# Export
fig.savefig('iptm_scatter.pdf', dpi=300, bbox_inches='tight')
fig.savefig('iptm_scatter.png', dpi=300, bbox_inches='tight')


plt.show()


# # ---------- Plot ----------
# fig, ax = plt.subplots(figsize=(6, 5))


# positions = [1, 2]  # x-positions for the two groups


# # Scatter only (jittered so points do not overlap)
# np.random.seed(42)  # reproducible jitter
# for i, (vec, label, color) in enumerate(zip(
#         [binder, non_binder],
#         ["Binder", "Non binder"],
#         ["purple", "orange"])):
#     x = np.random.normal(positions[i], 0.1, size=len(vec))
#     ax.scatter(x, vec, alpha=0.7, s=30, color=color, edgecolor=color, label=label)


# # Calculate means and standard errors
# means = [np.mean(binder), np.mean(non_binder)]
# stds  = [np.std(binder, ddof=1), np.std(non_binder, ddof=1)]
# ns    = [len(binder), len(non_binder)]
# sems  = [s / np.sqrt(n) for s, n in zip(stds, ns)]


# # Add mean lines and error bars
# for i, (mean_val, sem, color) in enumerate(zip(means, sems, ["purple", "orange"])):
#     # Horizontal line at the mean
#     ax.hlines(mean_val, positions[i] - 0.2, positions[i] + 0.2,
#               color=color, linewidth=2)
#     # Error bars (vertical lines for SEM)
#     ax.errorbar(positions[i], mean_val, yerr=sem,
#                 color=color, capsize=5, capthick=2, elinewidth=2)


# # Axis styling (fontsize 10pt, no x-label)
# ax.set_xticks(positions)
# ax.set_xticklabels(["Binder", "Non Binder"])
# ax.set_ylabel('IPTM Alphafold3', fontsize=16)
# ax.tick_params(axis='both', labelsize=16)
# #ax.legend(fontsize=8)


# plt.tight_layout()


# # Export
# fig.savefig('iptm_scatter.pdf', dpi=300, bbox_inches='tight')
# fig.savefig('iptm_scatter.png', dpi=300, bbox_inches='tight')


# plt.show()


# # ---------- Plot ----------
# fig, ax = plt.subplots(figsize=(6, 5))


# positions = [1, 2]  # x-positions for the two groups


# # Boxplot
# ax.boxplot([binder, non_binder],
#            positions=positions,
#            widths=0.6,
#            patch_artist=True,
#            boxprops=dict(facecolor='lightgray', color='black'),
#            medianprops=dict(color='red', linewidth=2))


# # Overlaid scatter (jittered so points do not overlap)
# np.random.seed(42)  # reproducible jitter
# for i, (vec, label) in enumerate(zip([binder, non_binder],
#                                       ["Binder", "Non binder"])):
#     x = np.random.normal(positions[i], 0.1, size=len(vec))
#     ax.scatter(x, vec, alpha=0.6, s=30, edgecolor='black', color='white', label=label)


# # Axis labels & styling (fontsize 10pt as requested)
# ax.set_xticks(positions)
# ax.set_xticklabels(["Binder", "Non binder"])
# #ax.set_xlabel('Class', fontsize=10)
# ax.set_ylabel('iptm value', fontsize=10)
# ax.tick_params(axis='both', labelsize=10)
# ax.legend(fontsize=8)


# plt.tight_layout()


# # Export
# fig.savefig('iptm_boxplot_scatter.pdf', dpi=300, bbox_inches='tight')
# fig.savefig('iptm_boxplot_scatter.png', dpi=300, bbox_inches='tight')


# plt.show()
