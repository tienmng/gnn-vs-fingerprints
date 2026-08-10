# When does a graph neural network actually beat molecular fingerprints?

A controlled comparison of a GINE message-passing network against a Morgan-fingerprint +
descriptor LightGBM baseline on MoleculeNet datasets, run under **both** random and
Bemis–Murcko scaffold splits, with 3 seeds per configuration.

The question is not "can I train a GNN". It is: *on datasets of the size most drug-discovery
teams actually have, does a learned representation beat a hand-crafted one — and how much of
any apparent win is just leakage from a random split?*

<!-- FILL IN after running: paste your one-sentence headline claim here.
     Example: "On ESOL (n=1128), the GNN wins by 6% under a random split and loses by 4%
     under a scaffold split — most of the apparent advantage is split leakage." -->

![headline](results/fig_esol.png)

---

## Results

<!-- FILL IN: paste the contents of results/table_esol.md here -->

| model | split | rmse (mean ± s.d.) |
|---|---|---|
| baseline | random | _run it_ |
| baseline | scaffold | _run it_ |
| gnn | random | _run it_ |
| gnn | scaffold | _run it_ |

Metrics are RMSE in log mol/L for ESOL/Lipophilicity, ROC-AUC for BBBP/BACE.
Mean ± sample standard deviation over 3 seeds. Test sets are never touched during
training or model selection; early stopping uses the validation split only.

**Scaffold overlap between train and test is asserted to be exactly 0** on every
scaffold-split run (`src/run_all.py`), so the comparison cannot be quietly leaking.

---

## Why the split matters

A random split puts close analogues of test molecules into the training set. Real projects
never work that way: you train on the series you have and predict on the series you are
about to make. The scaffold split approximates that, and the gap between the two panels of
the headline figure is the amount of performance that a random split invents for free.

<!-- FILL IN: 3-4 sentences on the size of the gap you measured. -->

## Applicability domain

Error against maximum Tanimoto similarity to any training molecule — the practical version
of "should I trust this prediction?".

![applicability domain](results/error_vs_sim_esol.png)

<!-- FILL IN: state the ratio, e.g. "MAE roughly doubles between the 0.7-1.0 and 0.0-0.3
     similarity bins" -->

## Error analysis

The 20 worst test predictions, drawn, with molecular weight / logP / TPSA / rotatable-bond
counts in `results/worst20_esol.csv`.

![worst predictions](results/worst20_esol.png)

<!-- FILL IN: what do the failures have in common? Typical findings on ESOL: large and
     flexible molecules, heavy halogenation, organometallics, and a handful of entries where
     the reference measurement itself looks suspect. -->

---

## What I expected vs. what happened

<!-- FILL IN, ~150 words, written honestly. This is the section people actually read.
     If the GNN lost, say so plainly and explain why you think it lost. -->

## Limitations

- Single-target datasets of ~1–4k molecules; conclusions do not transfer to the 10⁵–10⁶
  molecule regime where learned representations are expected to dominate.
- No hyperparameter search for either model beyond sensible defaults — a tuned GNN might
  close the gap, and a tuned LightGBM might widen it.
- Scaffold split is a proxy for prospective evaluation; a chronological split is closer to
  how a real programme unfolds.
- 2D topology only. No conformers, no 3D, no quantum-derived descriptors.
- Assay error in the reference labels is not modelled, and for some datasets it is a
  substantial fraction of the residual.

---

## Reproduce

```bash
git clone https://github.com/<you>/gnn-vs-fingerprints.git
cd gnn-vs-fingerprints
pip install -r requirements.txt

python -m src.run_all --dataset esol --seeds 0 1 2   # ~15 min on a free Colab T4
python -m src.analyze --dataset esol
```

Datasets download automatically to `data/` on first run
(`esol`, `lipo`, `bbbp`, `bace`). Everything is CPU-runnable; a T4 makes the GNN
roughly 8× faster.

Colab: open `notebooks/colab.ipynb` and run all cells.

## Repo layout

```
src/data.py      SMILES -> PyG graphs. Explicit atom/bond featuriser, no black boxes.
src/splits.py    random + Bemis-Murcko scaffold splits, with a leakage report
src/baseline.py  Morgan(2048, r=2) + 16 RDKit descriptors -> LightGBM
src/model.py     GINE message-passing network with residual connections
src/train.py     training loop, early stopping, metrics
src/run_all.py   the {model} x {split} x {seed} grid -> results/*.csv
src/analyze.py   figures, worst-prediction table, applicability domain
```

## Notes on design choices

- **Own featuriser rather than `MoleculeNet`'s 9 opaque features** — the atom and bond
  features are listed explicitly in `src/data.py` and can be defended line by line.
- **Target standardised with training statistics only.** Using whole-dataset mean/std is a
  small, invisible leak that shows up in a surprising number of public notebooks.
- **Sum pooling included alongside mean and max**, because solubility and lipophilicity are
  extensive-ish properties and mean pooling alone discards molecular size.
- **Early stopping on validation, evaluation on test, once.** No test-set peeking.

## License

MIT
