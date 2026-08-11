# When does a graph neural network actually beat molecular fingerprints?

**On ESOL (n=1,117), changing the split protocol moves RMSE by ~0.17 log units — an order of
magnitude more than the difference between a GINE message-passing network and a
Morgan-fingerprint LightGBM baseline (~0.01). Scaffold-split seed variance (±0.15) also
exceeds the model gap, so single-seed comparisons on this dataset cannot resolve
architecture differences.**

A controlled comparison of the two models under both random and Bemis–Murcko scaffold
splits, 3 seeds per configuration.

![applicability domain](results/error_vs_sim_esol.png)

## Takeaways

- **Protocol beats architecture.** Random → scaffold split costs 0.163–0.178 RMSE. Baseline
  → GNN changes it by 0.003–0.012.
- **The GNN does not beat fingerprints here.** 0.819 vs 0.816 on the scaffold split, at
  n=1,117. Statistically indistinguishable.
- **But the ranking inverts with novelty.** The GNN is better below 0.3 Tanimoto similarity
  to training, the baseline better above it. The aggregate metric hides this.
- **Scaffold-split seed variance (±0.15) exceeds the model gap**, so a single-seed comparison
  of these two architectures on ESOL cannot resolve anything.
- **Half the worst-case error is one scaffold family.** The splitter placed all monocyclic
  pyridines in the test set; ten of the twenty worst predictions are pyridines, all failing
  in the same direction.

---

## Results

ESOL (aqueous solubility, log mol/L), 1,117 clean unique molecules, 80/10/10 split.

| model | split | RMSE (mean ± s.d., 3 seeds) |
|---|---|---|
| baseline (Morgan + descriptors → LightGBM) | random | 0.653 ± 0.032 |
| baseline | scaffold | 0.816 ± 0.148 |
| GINE MPNN | random | 0.641 ± 0.067 |
| GINE MPNN | scaffold | 0.819 ± 0.168 |

| effect | size |
|---|---|
| split protocol (random → scaffold), baseline | +0.163 RMSE (+25%) |
| split protocol (random → scaffold), GNN | +0.178 RMSE (+28%) |
| model choice (baseline → GNN), random split | −0.012 RMSE |
| model choice (baseline → GNN), scaffold split | +0.003 RMSE |

![rmse by model and split](results/fig_esol.png)

Test sets are not touched during training or model selection; early stopping uses the
validation split only. Train/test scaffold overlap is asserted to be 0 on every
scaffold-split run (`src/run_all.py`). The random split leaks 22 shared scaffolds into the
test set.

## Split protocol vs. model choice

A random split places close analogues of test molecules in the training set. The measured
cost of removing that leak is 0.163–0.178 RMSE, about a quarter of total error. The
difference between architectures is 0.003–0.012, smaller than the seed-to-seed standard
deviation of either model under either split.

Seed variance rises from 0.032 to 0.148 (baseline) and 0.067 to 0.168 (GNN) under scaffold
splitting, because there are only 269 scaffolds and which ones land in test matters. A
single-seed scaffold-split number on ESOL carries roughly ±0.15 RMSE of noise, more than ten
times the model difference it would be used to demonstrate. Three seeds gives a crude s.d.
estimate; treat it as an order of magnitude.

## Applicability domain

Mean absolute error against maximum Tanimoto similarity to any training molecule.

| max Tanimoto to train | n | baseline MAE | GNN MAE |
|---|---|---|---|
| 0.0 – 0.3 | 30 | 1.132 | **0.906** |
| 0.3 – 0.4 | 25 | **0.696** | 0.857 |
| 0.4 – 0.5 | 29 | **0.752** | 0.826 |
| 0.5 – 0.7 | 18 | **0.557** | 0.782 |
| 0.7 – 1.0 | 11 | **0.449** | 0.532 |

The curves cross. The baseline degrades 2.5× from the most- to least-similar bin; the GNN
1.7×. The baseline is the better model above ~0.3 similarity and the worse model below it.

A Morgan fingerprint is a similarity-matching representation, so it performs well when the
test molecule resembles a training molecule and poorly when it does not. Averaged over the
full test set the two effects cancel, which is why the aggregate RMSE shows no difference.
The highest-similarity bin holds 11 molecules and is noisy; the 0.0–0.3 bin (n=30) is better
supported.

## Error analysis

![parity](results/parity_esol.png)

Predicted vs. measured on the scaffold-split test set, seed 0 only — so these RMSE values are
one draw, not the 3-seed means above. Both models compress the range: points sit above the
diagonal at the insoluble end and below it at the soluble end. The GNN extrapolates further
into the low tail, the baseline flattens against it.

The 20 worst test predictions, with descriptors in `results/worst20_esol.csv`.

![worst predictions](results/worst20_esol.png)

Seventeen of the twenty fall into two chemical families, failing in opposite directions.

**Ten substituted pyridines, all predicted too insoluble.** Nicotinamide, four lutidines,
2-ethylpyridine, 2- and 4-hydroxypyridine, isoniazid. Measured 0.0 to +1.0, predicted −0.7
to −1.7: a systematic offset of 1.6–2.3 log units in one direction. Three further
N-heterocycles (guanine, a dimethylxanthine diol, a chlorotriazine) fail the same way.

**Seven large lipophilic agrochemicals, predicted too soluble.** logP 4.3–6.5, MW 340–505 —
two pyrethroid esters, two benzoylureas, a hydroxycoumarin, an organophosphate phthalimide,
a diaryl ether. Measured −6.0 to −8.6, predicted −4.4 to −6.6.

The pyridine cluster follows from the split. Bemis–Murcko groups monocyclic azines under one
scaffold, so the splitter moved the whole family into the test set. No pyridine appears in
training, and the model applies the shrinkage learned from carbocyclic aromatics instead.
Half the worst-case error comes from one missing scaffold group — which is also why the
aggregate RMSE varies so much between seeds.

There is a representational limit underneath the second cluster. Aqueous solubility depends
on solid-state lattice energy as well as partitioning; the General Solubility Equation
approximates logS ≈ 0.5 − 0.01(MP − 25) − logP. Melting point is not a function of the 2D
molecular graph, so part of this error is not addressable by architecture changes on 2D
inputs.

## Practical reading

At this data scale the two models are equivalent on aggregate metrics, and the choice
between them depends on deployment. For triaging analogue series close to existing data, the
fingerprint baseline is better and much cheaper. For scoring novel chemotypes, the GNN
degrades more slowly and the aggregate metric does not show it.

## Limitations

- Single-target dataset of ~1.1k molecules; does not transfer to the 10⁵–10⁶ regime where
  learned representations are expected to dominate.
- 3 seeds gives a crude standard deviation estimate.
- No hyperparameter search for either model beyond defaults.
- Scaffold split is a proxy for prospective evaluation; a chronological split is closer to
  how a real programme unfolds.
- 2D topology only. No conformers, no 3D, no quantum-derived descriptors.
- Assay error in the ESOL reference values is not modelled and is non-trivial for the
  poorly-soluble tail.

## Reproduce

```bash
git clone https://github.com/tienmng/gnn-vs-fingerprints.git
cd gnn-vs-fingerprints
pip install -r requirements.txt

python -m src.run_all --dataset esol --seeds 0 1 2
python -m src.analyze --dataset esol
```

Datasets download to `data/` on first run (`esol`, `lipo`, `bbbp`, `bace`). CPU-runnable.
See `QUICKSTART.md` for Colab.

## Layout

```
src/data.py      SMILES -> PyG graphs; explicit atom/bond featuriser
src/splits.py    random + Bemis-Murcko scaffold splits, with a leakage report
src/baseline.py  Morgan(2048, r=2) + 16 RDKit descriptors -> LightGBM
src/model.py     GINE message-passing network with residual connections
src/train.py     training loop, early stopping, metrics
src/run_all.py   {model} x {split} x {seed} grid -> results/*.csv
src/analyze.py   figures, worst-prediction table, applicability domain
```

## Design notes

- 44 atom and 11 bond features, listed explicitly in `src/data.py`, rather than
  `MoleculeNet`'s 9 preset features.
- Target standardised with training statistics only.
- Sum pooling alongside mean and max, since solubility scales with molecular size.
- Early stopping on validation, evaluation on test once.

## License

MIT
