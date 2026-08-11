# Quickstart

## Colab

Runtime -> Change runtime type -> T4 GPU. Open `notebooks/colab.ipynb` and run all,
or paste:

```python
!pip -q install rdkit lightgbm torch_geometric
!git clone -q https://github.com/tienmng/gnn-vs-fingerprints.git
%cd gnn-vs-fingerprints
!python -m src.run_all --dataset esol --seeds 0 1 2
!python -m src.analyze --dataset esol
```

Save and download:

```python
!zip -qr results.zip results
from google.colab import files
files.download("results.zip")
```

Unzip `results/` into your local clone before committing.

## Local

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all --dataset esol --seeds 0 1 2
python -m src.analyze --dataset esol
```

Smoke test:

```bash
python -m src.run_all --dataset esol --seeds 0 --splits scaffold --epochs 20
```

## Learning curve

Test error vs. training-set size, scaffold split:

```bash
python -m src.learning_curve --dataset lipo --seeds 0 1 2
```

Writes `results/curve_lipo.csv` and `results/curve_lipo.png`. Use `--sizes` to override
the default log grid.

## Cross-dataset summary

After running the above on more than one dataset:

```bash
python -m src.summarize
```

Reads every `results/runs_*.csv` and `results/curve_*.csv` present and writes
`results/fig_all_curves.png`, `results/fig_all_splits.png` and `results/summary_all.md`.

## Options

```
--dataset   esol | lipo | bbbp | bace
--seeds     seeds to average over
--splits    random scaffold
--models    baseline gnn
--epochs    max epochs before early stopping
--csv       local CSV instead of downloading
```

## Commit

```bash
git add results README.md
git commit -m "Results"
git push
```
