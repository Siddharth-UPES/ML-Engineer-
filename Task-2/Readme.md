## Task 2: Model Debugging & Stability

### Problem
The initial model showed inconsistent performance across multiple runs due to randomness and improper preprocessing, leading to unstable evaluation results.

### Issues Identified
- Randomness in train-test splitting
- Data leakage caused by scaling before splitting the data
- Inconsistent preprocessing between training and testing

### Fixes Applied
- Fixed randomness using a constant `random_state`
- Applied feature scaling only on training data to prevent data leakage
- Reused the same scaler for test data and inference

### Result
After applying these fixes, the model produced stable and reproducible results across multiple runs with consistent evaluation metrics.

### Key Learning
Ensuring reproducibility and preventing data leakage are critical steps in building reliable and production-ready machine learning systems.
