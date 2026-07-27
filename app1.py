import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib


# PAGE CONFIGURATION

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)


# LOAD MACHINE LEARNING MODEL

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Columns that must be scaled with the SAME StandardScaler used in training
numeric_columns = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p",
    "academic_average"
]

model_features = [
    'ssc_p',
    'hsc_p',
    'degree_p',
    'etest_p',
    'mba_p',
    'academic_average',
    'gender_M',
    'ssc_b_Others',
    'hsc_b_Others',
    'hsc_s_Commerce',
    'hsc_s_Science',
    'degree_t_Others',
    'degree_t_Sci&Tech',
    'workex_Yes',
    'specialisation_Mkt&HR'
]

st.success("✅ Machine Learning model loaded successfully!")


FALLBACK_STATUS_COUNTS = {"Placed": 148, "Not Placed": 67}

raw_score_columns = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]

FALLBACK_GROUP_MEANS = pd.DataFrame({
    "ssc_p":    {"Not Placed": 57.54, "Placed": 71.72},
    "hsc_p":    {"Not Placed": 58.40, "Placed": 69.93},
    "degree_p": {"Not Placed": 61.13, "Placed": 68.74},
    "etest_p":  {"Not Placed": 69.59, "Placed": 73.24},
    "mba_p":    {"Not Placed": 61.61, "Placed": 62.58},
})

DATA_PATH = "data/Placement_Data_Full_Class1.csv"

if os.path.exists(DATA_PATH):
    _df = pd.read_csv(DATA_PATH)
    status_counts = _df["status"].value_counts()
    group_means = _df.groupby("status")[raw_score_columns].mean().round(2)
else:
    status_counts = pd.Series(FALLBACK_STATUS_COUNTS)
    group_means = FALLBACK_GROUP_MEANS

placed_avg = group_means.loc["Placed"].to_dict()

SCORE_LABELS = {
    "ssc_p": "10th (SSC) percentage",
    "hsc_p": "12th (HSC) percentage",
    "degree_p": "Degree percentage",
    "etest_p": "Employability test score",
    "mba_p": "MBA percentage",
}


def generate_recommendations(prediction, scores, workex):
    """Build a short list of tailored, data-driven suggestions by
    comparing the student's scores against the average scores of
    students who were historically placed."""

    recommendations = []
    gaps = {col: placed_avg[col] - scores[col] for col in raw_score_columns}

    if prediction == 1:

        recommendations.append(
            "Strong profile overall — keep sharpening your resume, "
            "portfolio, and interview skills to convert this into offers."
        )

        weakest = max(gaps, key=gaps.get)

        if gaps[weakest] > 3:
            recommendations.append(
                f"Your {SCORE_LABELS[weakest]} is a little below the "
                f"average of placed students (by about {gaps[weakest]:.1f} "
                "points) — polishing this further would make your profile "
                "even stronger."
            )

        if workex == "No":
            recommendations.append(
                "Adding an internship or a substantial project before "
                "final placements can further boost your competitiveness."
            )

    else:

        recommendations.append(
            "This isn't a fixed outcome — placement chances can improve "
            "a lot with focused preparation in the right areas."
        )

        weak_areas = sorted(
            (col for col, gap in gaps.items() if gap > 0),
            key=lambda col: gaps[col],
            reverse=True
        )

        for area in weak_areas[:3]:
            recommendations.append(
                f"{SCORE_LABELS[area]} is below the average of placed "
                f"students by about {gaps[area]:.1f} points — this is "
                "worth prioritizing."
            )

        if workex == "No":
            recommendations.append(
                "Work experience makes a noticeable difference in this "
                "dataset — consider taking up an internship, part-time "
                "role, or a substantial hands-on project."
            )

        recommendations.append(
            "Mock interviews and aptitude-test practice are also worth "
            "investing time in, since employability-test performance "
            "matters for placement outcomes."
        )

    return recommendations

# MAIN TITLE

st.title("🎓 Student Placement Prediction System")

st.write(
    "This application predicts whether a student is likely "
    "to be placed based on academic and employability factors."
)

# INFORMATION

st.info(
    "Welcome! Enter the student's details below to get "
    "a placement prediction."
)

# SIDEBAR

st.sidebar.title("🎓 Placement Predictor")

st.sidebar.write(
    "Student Placement Prediction System"
)

st.sidebar.metric(
    "Historical Placement Rate",
    f"{(status_counts.get('Placed', 0) / status_counts.sum() * 100):.1f}%"
)


# STUDENT INPUT FORM

st.header("👤 Student Information")


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )


with col2:

    ssc_p = st.number_input(
        "SSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )


with col3:

    ssc_b = st.selectbox(
        "SSC Board",
        [
            "Central",
            "Others"
        ]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    hsc_p = st.number_input(
        "HSC Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )


with col2:

    hsc_b = st.selectbox(
        "HSC Board",
        [
            "Central",
            "Others"
        ]
    )


with col3:

    hsc_s = st.selectbox(
        "HSC Specialisation",
        [
            "Science",
            "Commerce",
            "Arts"
        ]
    )


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    degree_p = st.number_input(
        "Degree Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )


with col2:

    degree_t = st.selectbox(
        "Degree Type",
        [
            "Sci&Tech",
            "Comm&Mgmt",
            "Others"
        ]
    )


with col3:

    workex = st.selectbox(
        "Work Experience",
        [
            "Yes",
            "No"
        ]
    )


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    etest_p = st.number_input(
        "Employability Test Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )


with col2:

    specialisation = st.selectbox(
        "MBA Specialisation",
        [
            "Mkt&HR",
            "Mkt&Fin"
        ]
    )


with col3:

    mba_p = st.number_input(
        "MBA Percentage",
        min_value=0.0,
        max_value=100.0,
        value=60.0
    )

# PREDICT BUTTON

st.markdown("---")

predict_button = st.button(
    "🔮 Predict Placement",
    use_container_width=True
)


if predict_button:

    # Create DataFrame using student input values
    academic_average = (
        ssc_p +
        hsc_p +
        degree_p
    ) / 3

    student_encoded = pd.DataFrame({
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'etest_p': [etest_p],
        'mba_p': [mba_p],

        'academic_average': [academic_average],

        # Gender
        'gender_M': [1 if gender == "M" else 0],

        # SSC Board
        'ssc_b_Others': [1 if ssc_b == "Others" else 0],

        # HSC Board
        'hsc_b_Others': [1 if hsc_b == "Others" else 0],

        # HSC Specialisation
        'hsc_s_Commerce': [1 if hsc_s == "Commerce" else 0],
        'hsc_s_Science': [1 if hsc_s == "Science" else 0],

        # Degree Type
        'degree_t_Others': [1 if degree_t == "Others" else 0],
        'degree_t_Sci&Tech': [1 if degree_t == "Sci&Tech" else 0],

        # Work Experience
        'workex_Yes': [1 if workex == "Yes" else 0],

        # MBA Specialisation
        'specialisation_Mkt&HR': [
            1 if specialisation == "Mkt&HR" else 0
        ]
    })

    student_encoded = student_encoded[model_features]

    # Display success message
    st.success(
        "✅ Student data created successfully!"
    )

    # # Display DataFrame
    # st.subheader(
    #     "📋 Student Data"
    # )

    # st.dataframe(
    #     student_encoded
    # )

    # MATCH MODEL FEATURES

    student_encoded = student_encoded.reindex(
        columns=model_features,
        fill_value=0
    )

    student_encoded[numeric_columns] = scaler.transform(
        student_encoded[numeric_columns]
    )

    # DISPLAY FINAL DATA   

    # st.subheader(
    #     "🤖 Final Data Sent to Machine Learning Model (scaled)"
    # )

    # st.dataframe(
    #     student_encoded
    # )

    # MAKE PREDICTION

    prediction = model.predict(
        student_encoded
    )[0]

    prediction_probability = model.predict_proba(
        student_encoded
    )[0][1]

    # DISPLAY PREDICTION

    st.subheader(
        "🎯 Placement Prediction"
    )

    if prediction == 1:

        st.success(
            f"🎉 Prediction: PLACED (Probability: {prediction_probability * 100:.2f}%)"
        )

    else:

        st.error(
            f"⚠️ Prediction: NOT PLACED (Probability of placement: {prediction_probability * 100:.2f}%)"
        )

    # ============================================================
    # SUGGESTIONS & RECOMMENDATIONS
    # ============================================================

    student_scores = {
        "ssc_p": ssc_p,
        "hsc_p": hsc_p,
        "degree_p": degree_p,
        "etest_p": etest_p,
        "mba_p": mba_p,
    }

    recommendations = generate_recommendations(
        prediction=prediction,
        scores=student_scores,
        workex=workex
    )

    st.subheader("💡 Suggestions & Recommendations")

    for rec in recommendations:
        st.markdown(f"- {rec}")

# MODEL FEATURE INFORMATION

st.subheader(
    "🔍 Model Feature Information"
)

if hasattr(model, "feature_names_in_"):

    st.write(
        "Features expected by the model:"
    )

    st.write(
        list(model.feature_names_in_)
    )

else:

    st.warning(
        "⚠️ The saved model does not contain feature names."
    )


# FOOTER

st.markdown("---")

st.caption(
    "🎓 Student Placement Prediction System | "
    "Machine Learning Project using Logistic Regression"
)