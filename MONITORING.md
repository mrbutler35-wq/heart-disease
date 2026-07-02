
# Production Data Drift Analysis

### 1. Which features showed drift and why?

The features that demonstrated significant, statistically relevant drift were **`age`** and **`trestbps`** (Resting Blood Pressure).

* **Why:** This drift occurred because our simulated production data mirrored an older patient subpopulation entering the clinical intake pipeline over time. Because age correlates biologically with vascular system stiffening, both distributions shifted substantially higher relative to the uniform baseline distributions captured in our historical `heart.csv` file.

### 2. Would this drift likely affect model performance?

**Yes, absolutely.** Linear or tree-based classification partitions learned during training depend closely on feature target boundaries. Because the structural values for `age` and `trestbps` shifted higher without matching alterations in label characteristics, the model will struggle with calibration error and face increased false negative rates, misclassifying high-risk older profiles as low risk.

### 3. What action would you recommend?

* **Recommendation: Investigate and Retrain.** Because the overall drift share crossed our configured threshold gate (25%), we should not continue running raw predictions unadjusted. I recommend engineering updated metadata groups reflecting this older patient population, combining the historical data matrix with new production inputs, and executing an automated retraining pipeline run to adjust the `RandomForestClassifier` boundaries.
  git commit -m "chore: remove raw data from git tracking to comply with DVC rules"git commit -m "chore: remove raw data from git tracking to comply with DVC rules"git commit -m "chore: remove raw data from git tracking to comply with DVC rules"
