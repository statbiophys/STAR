# GFRAL-DTA antibody discovery: figures

Scripts and data for the figures of the GFRAL-DTA paper. The STAR method itself lives at the
repository root (`Pipeline.ipynb`, `all_class/`); this directory only uses its output.

## Running

```bash
pip install -r ../requirements.txt
cd GFRAL-paper-figures
python Figure2_3_4.py
python Figure_3C.py
python Figure_5C.py
jupyter nbconvert --to notebook --execute Figure.ipynb
```

Figures are written to `figures/`. Run `Figure2_3_4.py` first: `Figure_5C.py` reads a table
it writes. Paths are resolved relative to the script, so the working directory does not
matter; `STAR_DATA` and `STAR_FIGURES` override the input and output roots. `Figure2_3_4.py`
uses roughly 3-4 GB of memory and a couple of minutes.

## Which script makes which figure

| Figure | Script |
|---|---|
| 2A, 2B, 2C, 3A, 3B, 4A, 4B, 4C, 4E | `Figure2_3_4.py` |
| 3C | `Figure_3C.py` |
| 4D, 5D, 5E | `Figure.ipynb` (reads `../data_test/`) |
| 5C | `Figure_5C.py` |
| 1, 5A | not script-generated (cartoon; PyMOL rendering of AlphaFold 3 models) |
| S3, S4 | HILARy, RAxML and iTOL, outside this repository |

## Data

```
data/
  single_cell/   Positive.tsv  Negative.tsv  total_mice_aligned.tsv  all_single_cell.tsv
                 all_single_cell_heavy_aligned.tsv  all_single_cell_paired.tsv
                 df_GFRAL.tsv  GFRAL_tag.xlsx  sequence_iptm.tsv
  bulk/          mouse_{1..5}/<DAY>/df_read.tsv
  spr/           SI_Dataset_1_withnumbers.xlsx
```

`data/README.md` describes every file and column. The bulk repertoires these tables are
derived from are on Zenodo: https://zenodo.org/records/22711596
