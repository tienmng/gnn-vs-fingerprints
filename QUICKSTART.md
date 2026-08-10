# Quickstart — from zero to a pushed repo

## A. Run it (Colab, ~25 min)

1. https://colab.research.google.com → **Runtime → Change runtime type → T4 GPU**
2. Upload `notebooks/colab.ipynb`, or paste these cells:

```python
!pip -q install rdkit lightgbm torch_geometric
!git clone -q https://github.com/<you>/gnn-vs-fingerprints.git
%cd gnn-vs-fingerprints
!python -m src.run_all --dataset esol --seeds 0 1 2
!python -m src.analyze --dataset esol
```

3. Download `results/` and drop it into your local clone.

Before pushing the repo you can develop against a local copy: zip `src/`, upload it to
the Colab file pane, and skip the `git clone` cell.

## B. Run it locally (CPU is fine, ~30 min)

```bash
cd gnn-vs-fingerprints
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_all --dataset esol --seeds 0 1 2
python -m src.analyze --dataset esol
```

Faster smoke test first (~2 min), to confirm everything imports and runs:

```bash
python -m src.run_all --dataset esol --seeds 0 --splits scaffold --epochs 20
```

## C. Fill in the README

`src/analyze.py` writes `results/table_esol.md`. Paste it into the Results section,
then fill in every `<!-- FILL IN -->` block. There are six. Budget 40 minutes and
do not exceed it.

## D. Push

```bash
cd gnn-vs-fingerprints
git init -b main
git add .
git commit -m "GNN vs fingerprints: random vs scaffold split comparison"
gh repo create gnn-vs-fingerprints --public --source=. --push
```

No `gh` CLI? Create the empty repo on github.com, then:

```bash
git remote add origin https://github.com/<you>/gnn-vs-fingerprints.git
git push -u origin main
```

Commit `results/` — the CSVs and figures are the deliverable, not the code.

## E. Then

Add the repo link to your CV and LinkedIn with the one-sentence result, not the title.
"Measured how much of a GNN's apparent advantage over fingerprints is an artefact of
random splitting" beats "Built a graph neural network for molecular property prediction".
