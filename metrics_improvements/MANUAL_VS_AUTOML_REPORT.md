# Manual Grid Search vs AutoML - Time-to-Target Report

This report compares a **manual hyperparameter grid search** (the old way: no MLflow, no AutoML, just nested loops) against the project's **AutoML** candidate set on the same train/test split.

## Summary

| Approach | Combos Tried | Wall Clock | Best RMSE | Reached Target | Time To Target | Combos To Target |
| --- | --- | --- | --- | --- | --- | --- |
| Manual Grid Search (no MLflow / no AutoML) | 22 | 6.71m | 8.3146 | True | 6.70m | 20 |
| AutoML (ml605_pipeline.automl) | 5 | 2.90m | 8.3966 | True | 2.87m | 5 |

**Conclusion:** AutoML reached RMSE <= 8.3966 in `2.87m` vs `6.70m` for the manual grid search - a time saving of `229.4s` (57.10%).

## Winning configuration per approach

- **Manual Grid Search (no MLflow / no AutoML)** - family=`ridge`, params=`{'alpha': 10.0}`
- **AutoML (ml605_pipeline.automl)** - family=`ridge_baseline`, params=`{'alpha': 1.0, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'positive': False, 'random_state': None, 'solver': 'auto', 'tol': 0.0001}`
