# STAR
Single Time-point Antibody Recognition
# STAR: Single Time-point Antibody Recognition

**STAR** (Single Time-point Antibody Recognition) is a computational pipeline designed to analyze antibody repertoires from a single time point. It enables clonal assignment and antigen specificity prediction from bulk B cell receptor (BCR) data using probabilistic modeling and machine learning.

---

## Features

- 🔍 **Specificity Prediction**: Infers antigen-specific sequences from background using statistical signals.
- 📦 **OLGA Integration**: Uses generative models of BCR recombination via OLGA.

---

## Repository Structure

- `Pipeline.ipynb`: Main Jupyter notebook walking through the full STAR pipeline.
- `all_class/`: Custom Python classes used by the notebook.
- `data_test/`: Sample datasets for testing the pipeline.
- `olga/`: Directory containing the OLGA model used for generation probability.
- `output/`: Folder where STAR saves analysis results.

---

## Getting Started

### Prerequisites

You will need:

- Python 3.6+
- Jupyter Notebook
- The following Python libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `olga` (from https://github.com/zsethna/OLGA)

### Installation

```bash
git clone https://github.com/statbiophys/STAR.git
cd STAR

