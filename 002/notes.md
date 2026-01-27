Observation

Linear Regression achieved higher test performance than a shallow Random Forest.
Random Forest underperformed despite being a nonlinear model.
Train and test metrics for Linear Regression were closely aligned.
Prediction errors increased at extreme solubility values.

Interpretation

The underlying feature–target relationship in the dataset is predominantly linear.
Increased model expressiveness does not guarantee better generalization under constrained capacity.
Model performance metrics can appear stable while hiding systematic errors at distribution tails.
Generalization metrics alone are insufficient to assess deployment reliability.

Stress Signal Identified

Model reliability degrades for out-of-distribution or extreme target values.
Nonlinear models may silently underfit when capacity constraints are imposed.
Baseline models define a trust envelope that more complex models must exceed to justify deployment.

Limitations

Single dataset
Limited hyperparameter exploration
No explicit distribution shift or noise injection
