# Fairness in AI Credit Risk Assessment

Banks increasingly use machine learning to decide who gets credit. If the training data reflects past bias, the model can repeat it. This project predicts credit risk on the German Credit dataset, measures whether the models treat **female and male applicants** differently, and reduces the gap with **reweighing**.

## Dataset

- [German Credit Dataset](https://www.kaggle.com/datasets/jumpingdino/german-credit-dataset) (Kaggle): **1,000** loan applicants, labelled good (1) or bad (0) credit risk
- 70% good and 30% bad credit risk
- 69% male and 31% female applicants. Gender was taken from the "status and sex" column and used as the protected attribute
- **Already in the data:** good credit rate of 0.723 for men and 0.648 for women, and 0.741 for applicants aged 30+ and 0.631 for younger ones

## Approach

1. **Exploratory analysis:** target, gender and age group distributions, and good credit rate per group
2. **Preprocessing:** scaling for numeric features and one-hot encoding for categorical ones, in a scikit-learn pipeline. Gender, status and sex, and age group were removed from the model's inputs to prevent leakage
3. **Models:** Logistic Regression, Decision Tree and Random Forest (800 training and 200 test applicants)
4. **Fairness metrics** (female compared with male applicants):
   - **Disparate Impact:** the female positive prediction rate divided by the male rate. Values close to 1 mean similar treatment
   - **Disparate Treatment:** the absolute difference between the male and female positive prediction rates
5. **Additional analysis:** confusion matrices, cosine similarity between applicant profiles, and association rule mining (Apriori)
6. **Mitigation:** reweighing, implemented from scratch. Training samples are weighted so that under-represented gender–label combinations count more, and the Decision Tree is retrained with these weights

## Results

**Baseline models**

| Model | Accuracy | F1 Score | Disparate Impact | Disparate Treatment |
|---|---|---|---|---|
| Logistic Regression | 0.700 | 0.787234 | 0.947195 | 0.038095 |
| Decision Tree | 0.685 | 0.797428 | 0.884409 | 0.102381 |
| Random Forest | 0.735 | 0.837920 | 1.023077 | 0.021429 |

The Decision Tree showed the largest gap between women and men, so it was chosen for mitigation.

**Decision Tree before and after reweighing**

| Stage | Accuracy | F1 Score | Disparate Impact | Disparate Treatment |
|---|---|---|---|---|
| Before Mitigation | 0.685 | 0.797428 | 0.884409 | 0.102381 |
| **After Mitigation** | **0.730** | **0.812500** | **0.924528** | **0.057143** |

## Key findings

- **Good performance doesn't guarantee fairness.** The Decision Tree had a strong F1 score but the largest gap between women and men.
- **Fairness and accuracy improved together.** Reweighing reduced Disparate Treatment from 0.102 to 0.057 while accuracy rose from 0.685 to 0.730.
- **Mitigation reduced the bias but didn't remove it.** Some gender disparity remains after reweighing.
- **Removing gender from the inputs wasn't enough.** The model never saw gender directly, yet still treated the groups differently.

## Limitations and next steps

- With only 1,000 applicants, subgroup fairness results are unstable. Cross-validation would make them more reliable
- Only gender was analysed. Age should also be checked, since younger applicants already have lower good credit rates in the data
- Other mitigation methods, such as threshold adjustment per group, could be compared with reweighing

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · mlxtend · Jupyter

## How to run

1. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib mlxtend jupyter
   ```
2. Download `german_credit_data.csv` from Kaggle and put it in the project folder.
3. Open `Ethical Issues in AI.ipynb` and run all cells.
