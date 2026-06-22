# Three-Body Problem — Trajectory Prediction with Machine Learning

Predicting the positions of three gravitationally interacting bodies over time using classical ML models, trained on 5,000 simulated trajectories.

This was a competitive assignment for the MSc in Big Data Analysis and Engineering (NOVA University of Lisbon), structured as an internal Kaggle-style challenge evaluated on a held-out test set.

---

## Problem

Given the initial positions and velocities of three bodies and a sequence of timesteps, predict their future (x, y) positions. The dataset contains 1,285,000 rows across 5,000 trajectories, each with 257 timesteps. Some trajectories end in collisions, which must be detected and excluded from training.

**Input features:** initial positions (x₀, y₀) of each body + timestep `t`  
**Targets:** positions (x, y) of each of the 3 bodies at time `t`

---

## Approach

### Data Pipeline
- Trajectory-aware train/validation/test split (70/15/15) — split by trajectory ID, not by row, to prevent data leakage
- Collision detection via zero-padding identification and trajectory-level exclusion
- Feature engineering: initial position extraction, distance features, interaction terms

### Models Explored (in order)
| Model | Validation RMSE |
|---|---|
| Linear Regression (baseline) | `[fill in]` |
| Polynomial Regression (degree `[d]`) | `[fill in]` |
| Ridge Regression (RidgeCV, best α) | `[fill in]` |
| k-Nearest Neighbors (best k) | `[fill in]` |
| KNN on reduced feature set | `[fill in]` |
| KNN on augmented feature set | `[fill in]` |

> Fill in the RMSE values from your notebook outputs before publishing.

### Key Design Decisions
- Trajectory-level splitting prevents leakage (rows from the same trajectory must not appear in both train and validation)
- Collision rows identified by zero-padding across all feature columns and excluded at the trajectory level
- Polynomial features combined with Ridge regularization to control overfitting on small sample fractions

---

## Stack

Python · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn