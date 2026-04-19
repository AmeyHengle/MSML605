# MLOps Hyperparameter Tuning - Time-to-Target Report

This report compares **without-MLflow tuning workflow** vs **MLflow-based tuning workflow** on how fast each reaches the same target metric.

| Workflow | Experiment | Metric Name | Target Threshold | Reached Target | Runs Needed | Best Metric | Time To Target | Absolute Time Saved Seconds | Percent Time Saved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Without MLflow Tuning (Baseline Workflow) | intensity-model-training | rmse | 40.0 | True | 1 | 10.9005 | 42.2s | -34.918 | -82.81% |
| With MLflow Tuning (AutoML Workflow) | intensity-model-automl | rmse | 40.0 | True | 1 | 8.3966 | 1.28m | -34.918 | -82.81% |

**Conclusion:** MLflow-based tuning saved `-34.9` seconds (-82.81%) to reach the same performance target.
