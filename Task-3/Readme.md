## Task 3: Model Performance Improvement

### Objective
Improve the model performance by at least 10% without changing the dataset or model architecture.

### Approach
Instead of retraining a new model, threshold tuning was applied on the predicted probabilities. By adjusting the decision threshold from the default value of 0.5 to 0.4, the model achieved better balance between precision and recall.

### Why Threshold Tuning?
- Fraud and anomaly detection problems benefit from optimized decision boundaries
- Allows performance improvement without increasing model complexity
- Suitable for production systems where retraining frequently is expensive

### Result
Threshold tuning resulted in a significant improvement in the F1-score, achieving more than 10% performance gain compared to the default threshold.

### Key Learning
Optimizing the decision threshold is an effective and lightweight technique to improve model performance in real-world ML systems.
