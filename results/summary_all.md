# Cross-dataset summary

All effects are paired within seed (and within model, for the split effect).
Positive split effect means the scaffold split is harder; positive model
effect means the GNN is better.

## Split effect

| dataset   | metric   |   baseline_scaffold |   split_effect |   paired_sd | pairs_scaffold_harder   |   pct_of_baseline |
|:----------|:---------|--------------------:|---------------:|------------:|:------------------------|------------------:|
| esol      | RMSE     |               0.816 |          0.171 |       0.134 | 6/6                     |              21   |
| lipo      | RMSE     |               0.707 |          0.075 |       0.08  | 5/6                     |              10.6 |
| bace      | ROC_AUC  |               0.877 |          0.019 |       0.057 | 2/6                     |               2.1 |
| bbbp      | ROC_AUC  |               0.916 |         -0.024 |       0.029 | 1/6                     |              -2.6 |


## Model effect at full training size

| dataset   | metric   |   model_effect |   paired_sd | seeds_gnn_wins   |   pct_of_baseline |
|:----------|:---------|---------------:|------------:|:-----------------|------------------:|
| esol      | RMSE     |         -0.003 |       0.037 | 1/3              |              -0.3 |
| lipo      | RMSE     |          0.052 |       0.053 | 3/3              |               7.4 |
| bace      | ROC_AUC  |         -0.021 |       0.011 | 0/3              |              -2.4 |
| bbbp      | ROC_AUC  |          0.019 |       0.016 | 3/3              |               2.1 |


## Learning-curve crossovers

| dataset   | metric   |   n_max |   advantage_at_max | wins_at_max   | crossover                                        |
|:----------|:---------|--------:|-------------------:|:--------------|:-------------------------------------------------|
| esol      | RMSE     |     893 |              0.012 | 1/3           | no seed-consistent crossover (best 1/3 at n=250) |
| lipo      | RMSE     |    3360 |              0.056 | 2/3           | crossover between 250 and 500                    |
| bace      | ROC_AUC  |    1210 |             -0.074 | 1/3           | no seed-consistent crossover (best 1/3 at n=100) |
| bbbp      | ROC_AUC  |    1580 |              0.005 | 2/3           | crossover between 250 and 500                    |


## Paired GNN advantage by training size

| dataset   |   n_train |    mean |     sd |   wins |   n |
|:----------|----------:|--------:|-------:|-------:|----:|
| esol      |       100 | -0.1596 | 0.1294 |      0 |   3 |
| esol      |       250 | -0.0156 | 0.0274 |      1 |   3 |
| esol      |       500 |  0.0028 | 0.0855 |      1 |   3 |
| esol      |       893 |  0.0117 | 0.133  |      1 |   3 |
| lipo      |       100 | -0.197  | 0.0263 |      0 |   3 |
| lipo      |       250 | -0.0046 | 0.0321 |      2 |   3 |
| lipo      |       500 |  0.0998 | 0.0467 |      3 |   3 |
| lipo      |      1000 |  0.0738 | 0.0264 |      3 |   3 |
| lipo      |      2000 |  0.0255 | 0.0181 |      3 |   3 |
| lipo      |      3360 |  0.0555 | 0.0813 |      2 |   3 |
| bace      |       100 | -0.0404 | 0.0658 |      1 |   3 |
| bace      |       250 | -0.0442 | 0.0632 |      1 |   3 |
| bace      |       500 | -0.0549 | 0.0682 |      1 |   3 |
| bace      |      1000 | -0.0502 | 0.087  |      1 |   3 |
| bace      |      1210 | -0.0742 | 0.0906 |      1 |   3 |
| bbbp      |       100 | -0.0424 | 0.0108 |      0 |   3 |
| bbbp      |       250 | -0.0101 | 0.0496 |      2 |   3 |
| bbbp      |       500 |  0.0001 | 0.0205 |      2 |   3 |
| bbbp      |      1000 |  0.0201 | 0.0125 |      3 |   3 |
| bbbp      |      1580 |  0.005  | 0.0049 |      2 |   3 |


