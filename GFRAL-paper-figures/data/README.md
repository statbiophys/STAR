# Data

## `single_cell/` — paired single cell and annotation tables

| File | Description |
|---|---|
| `Positive.tsv` | Single cell sequences from the antigen-positive sorted splenocyte fraction (day 67). |
| `Negative.tsv` | The same for the antigen-negative fraction. |
| `total_mice_aligned.tsv` | All paired single cell sequences across the three mice, MiXCR-aligned, with `spe` (Positive / Negative) and per-region annotation. |
| `all_single_cell.tsv` | All single cell reads: sample tag (`0`) and sequence. |
| `all_single_cell_heavy_aligned.tsv` | Heavy chains only, MiXCR-aligned. Column `0` is the sample tag. |
| `all_single_cell_paired.tsv` | Paired heavy and light chains per cell, with the sample tag. |
| `df_GFRAL.tsv` | The 221 antibodies expressed and tested by SPR: clone name, heavy and light chain, heavy CDR3, and binding call (BINDER / WEAK BINDER / NON BINDER). |
| `GFRAL_tag.xlsx` | SPR metadata. `GDB Clone Name` joins to `df_GFRAL.tsv`; `KD (M)` is the dissociation constant. |
| `sequence_iptm.tsv` | AlphaFold 3 interface pTM score per heavy chain. |

Sample tags read `<mouseID>_<day>[_Spleen_Ag_pos|_Spleen_Ag_neg|_blood]_<cell barcode>`,
with mouse IDs 1152 = mouse 1, 368 = mouse 2, 1149 = mouse 3.

## `bulk/mouse_{1..5}/<DAY>/df_read.tsv` — processed bulk IgH repertoires

25 files. One row per unique heavy chain CDR3 per sample, with the STAR clustering
statistics already computed.

| Column | Meaning |
|---|---|
| `aaSeqCDR3` | heavy chain CDR3 |
| `size` | number of reads |
| `freq_count` | frequency within the sample |
| `day` | sample label (`D3`, `D17`, …, `D67`, `D67_spleen`) |
| `family` | single-linkage cluster id, linking at edit distance 1 |
| `molteplicità` | number of distinct nucleotide CDR3s coding this sequence |
| `nb_neighbours_real` | number of neighbours, i.e. sequences differing by one amino acid |
| `cdr3_len` | CDR3 length |
| `nb_freq` | neighbour count normalised by sample size |
| `hits` | called as a hit by STAR |
| `spe` | also seen in the single cell data |
| `mouse` | mouse the sample comes from |

Days: mouse_1 D17 D25 D39 D46 D60 D67 · mouse_2 D17 D25 D39 D46 D60 D67 D67_spleen ·
mouse_3 D3 D17 D25 D39 D60 D67 · mouse_4 D3 D17 D39 · mouse_5 D3 D17 D39.
`mouse_2/D17/df_read.tsv` is empty: that sample yielded no usable data.

## `spr/SI_Dataset_1_withnumbers.xlsx`

Full SPR results for all tested antibodies, with the `category` column
(`STAR` / `SC` / `bulk`) used for Fig. 3C.
