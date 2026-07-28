# 🎓 Student Placement Prediction System

A machine learning web application that predicts whether a student is likely to be placed
during campus recruitment, based on academic performance and employability factors, and
gives the student simple, actionable recommendations to improve their placement chances.

**Live app:** https://campus-placemenet-pridiction-debugabhii.streamlit.app/

---

## 1. Problem Statement

Campus placement is one of the most consequential outcomes of a student's academic
journey, yet most students have no clear, data-backed sense of how their academic record
and profile compare to students who have historically been placed. Career services teams
and students alike currently rely on informal, anecdotal advice ("keep your percentage
above X", "internships help") rather than evidence drawn from actual placement outcomes.

This project addresses that gap: given a student's academic scores (SSC, HSC, degree, MBA,
employability test), stream, board, gender, and work experience, can we predict — before
placement season — whether that student is likely to be placed, and tell them *why*?

## 2. Business Objective

- **For students:** Provide an early, personalized signal of placement likelihood, along
  with specific areas to improve, so students can act on it during their remaining time on
  campus rather than after the fact.
- **For institutions / placement cells:** Identify at-risk students earlier in the academic
  cycle so training, mentoring, and interview-prep resources can be targeted where they
  will have the most impact, instead of being spread evenly across the whole batch.
- **Success criteria:** A model that predicts placement outcome with strong precision and
  recall on held-out data, packaged in an interactive tool that is usable by a
  non-technical student or placement coordinator with no ML background.

## 3. Dataset

- **Source:** Campus recruitment dataset (`Placement_Data_Full_Class1.csv`), 215 student
  records.
- **Target variable:** `status` — Placed / Not Placed (≈69% Placed, 31% Not Placed).
- **Features used:** SSC / HSC / Degree / MBA percentages, employability test score,
  academic average, gender, SSC & HSC board, HSC specialisation, degree type, work
  experience, MBA specialisation.
- `salary` and `sl_no` were dropped (salary is only known *after* placement, and would leak
  the outcome; `sl_no` is just a row identifier).

## 4. Methodology

1. **Cleaning:** dropped leakage/identifier columns, removed duplicate rows.
2. **Feature engineering:** added `academic_average` (mean of SSC, HSC, and Degree
   percentages); one-hot encoded all categorical columns (gender, boards, streams, degree
   type, work experience, specialisation).
3. **Train/test split:** 80/20, stratified on the target to preserve the class balance in
   both sets.
4. **Scaling:** `StandardScaler` fit on the training set only, applied to all numeric
   features (percentages + academic average) before training and before every prediction.
5. **Model:** `LogisticRegression` with `class_weight='balanced'`, to correct for the
   69/31 class imbalance in the dataset so the model doesn't default to predicting
   "Placed" for an average student.
6. **Deployment:** trained model and scaler serialized with `joblib` and served through a
   Streamlit web app (`app1.py`) that collects a student's details, applies the same
   preprocessing pipeline, and returns a prediction with a probability score.

## 5. Additional Requirement — Placement-Improvement Recommendations

Beyond a binary prediction, the app generates **simple, actionable recommendations** for
every student, derived directly from the model's findings:

- The trained model's coefficients show that **SSC percentage**, **work experience**, and
  **academic average** are the strongest positive predictors of placement, ahead of MBA
  percentage or employability test score.
- For each prediction, the app compares the student's five raw scores against the
  **average scores of historically placed students** and surfaces the largest gaps by
  name (e.g. *"12th (HSC) percentage is below the average of placed students by about 12
  points — this is worth prioritizing."*).
- If the student has no work experience, the app explicitly recommends taking up an
  internship or a substantial hands-on project, since work experience is one of the
  strongest drivers of placement in this dataset.
- Students already predicted "Placed" still get a suggestion (e.g. their weakest relative
  area, or an internship nudge) so the tool remains useful even for strong profiles.

This turns the tool from a pass/fail predictor into a lightweight advisory system.

## 6. Results

Final model performance on the held-out test set (20% of the data, 43 students):

| Metric | Score |
|---|---|
| Accuracy | 86.05% |
| Precision | 96.15% |
| Recall | 83.33% |
| F1-Score | 89.29% |

**Confusion Matrix**

| | Predicted Not Placed | Predicted Placed |
|---|---|---|
| **Actual Not Placed** | 12 | 1 |
| **Actual Placed** | 5 | 25 |

**Top predictive features** (by absolute logistic regression coefficient):

| Rank | Feature | Direction |
|---|---|---|
| 1 | SSC Percentage | ↑ increases placement likelihood |
| 2 | Work Experience | ↑ increases placement likelihood |
| 3 | Academic Average | ↑ increases placement likelihood |
| 4 | Degree Type: Sci & Tech | ↓ decreases placement likelihood |
| 5 | MBA Percentage | ↓ decreases placement likelihood (slightly) |

## 7. Images / Screenshots

> Add screenshots of the running app here before submitting. Suggested captures:

![Student Input Form](images/student_input_form.png)
*The student information form used to collect academic and employability details.*

![Placement Prediction Result](images/prediction_result.png)
*Prediction output showing the placement probability and status, along with the
generated recommendations.*

## 8. Project Structure

```
├── app1.py                              # Streamlit application
├── main.ipynb                           # Model training & evaluation notebook
├── requirements.txt                     # Python dependencies
├── logistic_regression_model.pkl        # Trained model
├── scaler.pkl                           # Fitted StandardScaler
├── data/
│   └── Placement_Data_Full_Class1.csv   # Training dataset
└── README.md
```

## 9. How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app1.py
```

The app will open at `http://localhost:8501`.

## 10. Tech Stack

- **Python**, **pandas** — data handling
- **scikit-learn** — model training (Logistic Regression, StandardScaler)
- **joblib** — model/scaler persistence
- **Streamlit** — web application and deployment (Streamlit Community Cloud)

## 11. Future Improvements

- Try additional models (Random Forest, Gradient Boosting) and compare against the
  logistic regression baseline.
- Collect more data to reduce reliance on `class_weight='balanced'` for handling imbalance.
- Track recommendation effectiveness over time (e.g. did students who improved their
  flagged weak area actually get placed in the next cycle).

## 12. Author

Abhijeet Verma — B.Tech CSE, ITS Engineering College
