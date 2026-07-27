# Student Placement Prediction 

## Files
- `main.ipynb` — the notebook, with fixes #3 and #4 applied to the relevant cells.
- `app1.py` — the Streamlit app, with fixes #1 and #2 applied.
- `data/Placement_Data_Full_Class1.csv` — your original dataset.
- `logistic_regression_model.pkl`, `scaler.pkl` — **already retrained** on your dataset
  with all four fixes applied. Ready to use as-is.

## How to run it
1. Keep `logistic_regression_model.pkl` and `scaler.pkl` in the same folder as `app1.py`.
2. Run:
   ```
   streamlit run app1.py
   ```
3. Check the "Placement Probability" shown after clicking Predict — it now moves
   meaningfully with the inputs instead of defaulting to "Placed".

Verified on the retrained model:
- All-60s / no work experience input → **Not Placed** (6.6% placement probability)
- Weak student (40s-50s, no work experience) → **Not Placed** (0.15%)
- Strong student (85-90s, work experience) → **Placed** (99.95%)

If you want to retrain from scratch yourself (e.g. after changing something), just re-run
`main.ipynb` top to bottom — it will overwrite the two `.pkl` files with fresh ones.

## If it's still biased after this
Try nudging the classification threshold in `app1.py` — e.g. only call it "Placed" when
`prediction_probability >= 0.55` or `0.6` instead of the default 0.5 — and see which
threshold best matches your own judgment of the borderline cases in your test set.
