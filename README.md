# Airline Passenger Satisfaction Analytics — Decision Tree Pipeline

## Project Overview
This project builds, optimizes, and evaluates a Decision Tree Classification model using Python and Scikit-Learn to predict passenger satisfaction based on airline survey records (`Invistico_Airline.csv`). By tuning structural tree hyperparameters via 5-fold cross-validation, the model achieves robust generalization, uncovers high-impact operational satisfaction drivers, and maps non-linear behavioral pathways.

---

## Technical Pipeline & Implementation Strategy

### 1. Data Cleaning & Preprocessing
* **Missing Value Handling:** Rows with incomplete data parameters are isolated and dropped to prevent optimization skewing.
* **Feature Encoding:** Categorical variables (e.g., travel purpose, class types) are converted using one-hot mapping (`pd.get_dummies`) with a dropped initial reference column to protect structural variance integrity.
* **Target Isolation:** The satisfaction target column is dynamically identified and cast into standard binary form (`1` for Satisfied, `0` for Dissatisfied).

### 2. Hyperparameter Optimization & Tuning
* **Cross-Validation Scheme:** The dataset is split into an 80/20 train/test layout. A comprehensive 5-fold `GridSearchCV` pipeline searches combinations of:
  * `max_depth`: `[3, 5, 7, 10]`
  * `min_samples_split`: `[2, 5, 10]`
* **Overfitting Safeguards:** Constraining tree depth ensures the final configuration generalizes cleanly to unseen test distributions, rather than memorizing noise.

### 3. Model Performance Visualizations
* **Decision Tree Topology Map:** The initial levels of the tree are rendered using `plot_tree` to expose structural logic paths.
* **Feature Importance Distribution:** Top-tier predictors are ranked and displayed via Seaborn charts to isolate key drivers of satisfaction.

---

## Strategic Business Interpretations

### Algorithmic Architecture Trade-offs
* **Decision Trees vs. Logistic Regression:** Logistic models assume strict, additive linear feature relationships. Conversely, Decision Trees act as flexible step-functions that discover complex, non-linear feature cross-interactions (e.g., the changing impact of `In-flight Wifi Quality` across different travel classes) without manual engineering.

### Top Operational Drivers
* Root-level splits prove that **In-flight Wifi Quality** and **Seat Comfort** dictate major customer sorting milestones. To scale satisfaction effectively, airline operations must prioritize cabin amenities over auxiliary service streams.

### Cost Configuration Matrix
* **False Positives (Danger Zone):** Misidentifying an unhappy customer as "Satisfied" causes severe retention churn, as they leave the platform without experiencing proactive outreach or loyalty recovery options.
* **False Negatives (Minor Overhead):** Classifying a satisfied passenger as "Dissatisfied" results only in negligible marketing overhead (e.g., sending an unnecessary retention coupon).
* **Strategic Objective:** Model classification thresholds are biased to favor high **Recall** metrics for the dissatisfied segment to minimize high-cost retention failures.

