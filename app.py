# =============================================================================
# HACKATHON TEAM DYNAMICS - PREDICTION WEB APPLICATION
# =============================================================================
# Professional Streamlit interface for predicting hackathon team performance
# using trained machine learning models.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

#st.set_option('deprecation.showfileUploaderEncoding', False)

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
MODEL_DIR = "models"
RESULTS_DIR = "results"

# -----------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------
st.set_page_config(
    page_title="Hackathon Team Dynamics Predictor",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="auto"
)

# -----------------------------------------------
# PROFESSIONAL STYLING (No emojis, clean design)
# -----------------------------------------------
st.markdown("""
<style>
    /* ---- Global Font ---- */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* ---- Hide Streamlit footer only, keep header for sidebar toggle ---- */
    footer {visibility: hidden;}

    /* ---- Fix mobile top spacing ---- */
    .block-container {
        padding-top: 1rem !important;
        max-width: 100% !important;
    }

    /* ---- Reduce top padding on all screens ---- */
    .stApp > header {
        height: 2.5rem !important;
    }

    /* ---- Make sidebar toggle always visible ---- */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        color: #1565C0 !important;
    }

    /* ---- Mobile responsive fixes ---- */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .main-title {
            font-size: 1.5rem !important;
        }
        .main-subtitle {
            font-size: 0.85rem !important;
        }
    }

    /* ---- Main Header ---- */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1565C0;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        border-bottom: 3px solid #1976D2;
        margin-bottom: 0.3rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* ---- Section Headers ---- */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1565C0;
        border-left: 4px solid #1976D2;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* ---- Prediction Result Cards ---- */
    .result-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card-success {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card-danger {
        background: #f8d7da;
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .result-label {
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ---- Info Box ---- */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1976D2;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #333;
    }

    /* ---- Sidebar Styling ---- */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1565C0;
        margin-bottom: 0.5rem;
    }

    /* ---- Table Styling ---- */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .styled-table th {
        background: #1565C0;
        color: white;
        padding: 10px 12px;
        text-align: left;
    }
    .styled-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #dee2e6;
    }
    .styled-table tr:nth-child(even) {
        background: #f8f9fa;
    }

    /* ---- Button ---- */
    .stButton > button {
        width: 100%;
        background: #1565C0;
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: #0D47A1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------
def check_models_exist():
    """Verify all required model files are present."""
    required_files = [
        'stacking_rf_gb_regressor.pkl',
        'stacking_rf_gb_classifier.pkl',
        'standard_scaler.pkl',
        'mentor_guidance_encoder.pkl',
        'won_hackathon_encoder.pkl'
    ]
    missing = []
    for f in required_files:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            missing.append(f)
    return missing


@st.cache_resource
def load_models():
    """Load all saved models and preprocessing artifacts."""
    reg_model = joblib.load(os.path.join(MODEL_DIR, 'stacking_rf_gb_regressor.pkl'))
    clf_model = joblib.load(os.path.join(MODEL_DIR, 'stacking_rf_gb_classifier.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
    feature_encoder = joblib.load(os.path.join(MODEL_DIR, 'mentor_guidance_encoder.pkl'))
    target_encoder = joblib.load(os.path.join(MODEL_DIR, 'won_hackathon_encoder.pkl'))
    return reg_model, clf_model, scaler, feature_encoder, target_encoder


def preprocess_input(input_data, scaler, feature_encoder):
    """Preprocess user input for prediction."""
    df = pd.DataFrame([input_data])
    df['mentor_guidance'] = feature_encoder.transform(
        df['mentor_guidance'].astype(str)
    )
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# -----------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">Navigation</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "Data Analysis",
        "Model Performance",
        "Methodology",
        "About"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#888;'>Batch 9 | Mini Project<br>"
    "Hackathon Team Dynamics</small>",
    unsafe_allow_html=True
)

# -----------------------------------------------
# SCROLL TO TOP WHEN PAGE CHANGES
# -----------------------------------------------
st.markdown("""
<script>
    // Scroll to top when page loads
    window.scrollTo(0, 0);
    
    // Also scroll parent container
    var mainContent = window.parent.document.querySelector('.main');
    if (mainContent) {
        mainContent.scrollTo(0, 0);
    }
</script>
""", unsafe_allow_html=True)

# Alternative scroll fix using Streamlit's built-in method
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = page

if st.session_state.previous_page != page:
    st.session_state.previous_page = page
    st.rerun()


# ===============================================
# PAGE 1: PREDICTION
# ===============================================
if page == "Prediction":

    # Header
    st.markdown(
        '<div class="main-title">Hackathon Team Dynamics Predictor</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Predict your hackathon team\'s performance using trained ML models'
        '</div>',
        unsafe_allow_html=True
    )

    # Check models
    missing = check_models_exist()
    if missing:
        st.error(f"Missing model files: {missing}. Run 'python train.py' first.")
        st.stop()

    # Load models
    reg_model, clf_model, scaler, feature_encoder, target_encoder = load_models()

    # Description
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Enter your hackathon team's details below.
        The system uses a Stacking Ensemble model (Random Forest + Gradient Boosting)
        to predict three outcomes: submission quality score, final rank, and
        whether your team is likely to win.
    </div>
    """, unsafe_allow_html=True)

    # ---- Input Section ----
    st.markdown(
        '<div class="section-header">Team Details</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        team_size = st.number_input(
            "Team Size",
            min_value=2, max_value=6, value=4, step=1,
            help="Number of members in your hackathon team (2 to 6)"
        )
        skill_diversity = st.slider(
            "Skill Diversity Score",
            min_value=0.0, max_value=1.0, value=0.75, step=0.01,
            help="Measures how diverse the skill sets are within the team. "
                 "0.0 = all members have same skills, "
                 "1.0 = highly diverse skill sets"
        )
        experience_level = st.number_input(
            "Average Experience Level",
            min_value=0, max_value=5, value=3, step=1,
            help="Average experience level of team members. "
                 "0 = complete beginner, 5 = expert level"
        )

    with col2:
        conflict_count = st.number_input(
            "Conflict Count",
            min_value=0, max_value=5, value=2, step=1,
            help="Number of significant disagreements or conflicts "
                 "within the team during the hackathon"
        )
        communication_rating = st.slider(
            "Communication Rating",
            min_value=1.0, max_value=5.0, value=4.0, step=0.1,
            help="Overall team communication quality rating. "
                 "1.0 = very poor communication, "
                 "5.0 = excellent communication"
        )

    with col3:
        hours_spent = st.number_input(
            "Hours Spent",
            min_value=5, max_value=80, value=35, step=1,
            help="Total number of hours the team spent working "
                 "on the hackathon project"
        )
        mentor_guidance = st.selectbox(
            "Mentor Guidance",
            options=["Yes", "No"],
            help="Did the team receive any mentorship or guidance "
                 "from an experienced person?"
        )

    st.markdown("---")

    # ---- Predict Button ----
    if st.button("Generate Prediction"):

        # Prepare input
        input_data = {
            'team_size': team_size,
            'skill_diversity_score': skill_diversity,
            'experience_level_avg': experience_level,
            'conflict_count': conflict_count,
            'communication_rating': communication_rating,
            'hours_spent': hours_spent,
            'mentor_guidance': mentor_guidance
        }

        # Preprocess and predict
        X_input = preprocess_input(input_data, scaler, feature_encoder)
        reg_pred = reg_model.predict(X_input)
        clf_pred = clf_model.predict(X_input)

        # Extract and clean values
        quality_score = float(np.clip(reg_pred[0][0], 0, 100))
        final_rank = int(max(1, round(reg_pred[0][1])))
        won = target_encoder.inverse_transform(clf_pred)[0]

        # ---- Display Results ----
        st.markdown(
            '<div class="section-header">Prediction Results</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Submission Quality Score</div>
                <div class="result-value">{quality_score:.1f}</div>
                <div class="result-label">out of 100</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Final Rank</div>
                <div class="result-value">#{final_rank}</div>
                <div class="result-label">position</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if won == "Yes":
                st.markdown(f"""
                <div class="result-card-success">
                    <div class="result-label">Hackathon Outcome</div>
                    <div class="result-value">WIN</div>
                    <div class="result-label">likely to win</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-danger">
                    <div class="result-label">Hackathon Outcome</div>
                    <div class="result-value">NO WIN</div>
                    <div class="result-label">unlikely to win</div>
                </div>
                """, unsafe_allow_html=True)

        # ---- Input Summary Table ----
        st.markdown("")
        st.markdown(
            '<div class="section-header">Input Summary</div>',
            unsafe_allow_html=True
        )
        summary_df = pd.DataFrame([input_data])
        summary_df.columns = [
            'Team Size', 'Skill Diversity', 'Experience Level',
            'Conflicts', 'Communication', 'Hours Spent', 'Mentor Guidance'
        ]
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ===============================================
# PAGE 2: DATA ANALYSIS (EDA)
# ===============================================
elif page == "Data Analysis":

    st.markdown(
        '<div class="main-title">Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Understanding patterns and distributions in the hackathon dataset'
        '</div>',
        unsafe_allow_html=True
    )

    # Explanation
    st.markdown("""
    <div class="info-box">
        <strong>What is EDA?</strong><br>
        Exploratory Data Analysis is the process of examining and visualizing
        the dataset before building models. It helps us understand data
        distributions, identify patterns, detect outliers, and discover
        relationships between features and targets. These insights guide
        our choice of preprocessing steps and model selection.
    </div>
    """, unsafe_allow_html=True)

    # EDA Plots
    st.markdown(
        '<div class="section-header">Distribution and Relationship Plots</div>',
        unsafe_allow_html=True
    )

    if os.path.exists('eda_plots.png'):
        st.image('eda_plots.png', use_container_width=True)

        # Explanation of each plot
        st.markdown("""
        <div class="info-box">
            <strong>Plot Descriptions:</strong><br><br>
            <strong>1. Distribution of Submission Quality Score</strong> —
            Shows how quality scores are spread across teams. A roughly
            uniform distribution means teams have varied performance levels.<br><br>
            <strong>2. Hours Spent vs Submission Quality</strong> —
            Examines whether spending more hours leads to better submissions.
            Points are colored by win status to show if winners follow a
            different pattern.<br><br>
            <strong>3. Mentor Guidance vs Winning</strong> —
            Compares how many teams won with vs without mentor guidance.
            Helps assess if mentorship is a significant factor.<br><br>
            <strong>4. Final Rank by Team Size</strong> —
            Box plots showing the distribution of final ranks for each
            team size. Helps identify if certain team sizes perform better.<br><br>
            <strong>5. Count of Won Hackathon</strong> —
            Shows the class balance between winning and non-winning teams.
            This revealed significant class imbalance in our dataset
            (approximately 85% No vs 15% Yes).
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("EDA plots not found. Run 'python train.py' first.")

    # Dataset Preview
    st.markdown(
        '<div class="section-header">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    if os.path.exists("Dataset/Hackathon_Team_Dynamics_Dataset.csv"):
        df = pd.read_csv("Dataset/Hackathon_Team_Dynamics_Dataset.csv")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 3)
        with col3:
            st.metric("Winners", df['won_hackathon'].value_counts().get('Yes', 0))
        with col4:
            st.metric("Non-Winners", df['won_hackathon'].value_counts().get('No', 0))

        st.markdown(
            '<div class="section-header">Sample Data (First 15 Rows)</div>',
            unsafe_allow_html=True
        )
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="section-header">Statistical Summary</div>',
            unsafe_allow_html=True
        )
        st.dataframe(df.describe(), use_container_width=True)


# ===============================================
# PAGE 3: MODEL PERFORMANCE
# ===============================================
elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance Analysis</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Comparing evaluation metrics across all trained models'
        '</div>',
        unsafe_allow_html=True
    )

    # Explanation
    st.markdown("""
    <div class="info-box">
        <strong>Why compare multiple models?</strong><br>
        Different algorithms learn patterns differently. By training and
        evaluating multiple models on the same data, we can identify which
        approach works best for our specific problem. We trained 4 regression
        models (predicting scores and ranks) and 4 classification models
        (predicting win/loss).
    </div>
    """, unsafe_allow_html=True)

    # Check if results folder has images
    if not os.path.exists(RESULTS_DIR) or len(os.listdir(RESULTS_DIR)) == 0:
        st.warning("No results found. Run 'python train.py' first.")
        st.stop()

    # ---- Classification Section ----
    st.markdown(
        '<div class="section-header">Classification Model Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <strong>Classification Task:</strong> Predict whether a team will
        win the hackathon (Yes/No).<br>
        <strong>Metrics Used:</strong><br>
        - <strong>Accuracy</strong> — Percentage of correct predictions overall<br>
        - <strong>Precision</strong> — Of all predicted "Yes", how many were actually "Yes"<br>
        - <strong>Recall</strong> — Of all actual "Yes", how many did we correctly predict<br>
        - <strong>F1-Score</strong> — Harmonic mean of Precision and Recall (balanced metric)
    </div>
    """, unsafe_allow_html=True)

    # Comparison chart
    comparison_path = os.path.join(RESULTS_DIR, 'model_performance_comparison.png')
    if os.path.exists(comparison_path):
        st.image(comparison_path, use_container_width=True)

    # Confusion Matrices
    st.markdown(
        '<div class="section-header">Confusion Matrices</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <strong>What is a Confusion Matrix?</strong><br>
        A table showing the counts of correct and incorrect predictions
        for each class. The diagonal shows correct predictions (True Positives
        and True Negatives). Off-diagonal values show errors (False Positives
        and False Negatives).
    </div>
    """, unsafe_allow_html=True)

    cm_files = sorted([f for f in os.listdir(RESULTS_DIR) if 'confusion_matrix' in f])
    if cm_files:
        cols = st.columns(2)
        for idx, f in enumerate(cm_files):
            with cols[idx % 2]:
                # Clean up filename for display
                display_name = f.replace('_confusion_matrix.png', '').replace('_', ' ').title()
                st.markdown(f"**{display_name}**")
                st.image(os.path.join(RESULTS_DIR, f), use_container_width=True)

    # ROC Curves
    roc_files = sorted([f for f in os.listdir(RESULTS_DIR) if 'roc_curve' in f])
    if roc_files:
        st.markdown(
            '<div class="section-header">ROC Curves</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="info-box">
            <strong>What is an ROC Curve?</strong><br>
            The Receiver Operating Characteristic curve plots True Positive Rate
            vs False Positive Rate at various classification thresholds.
            The Area Under the Curve (AUC) indicates how well the model
            distinguishes between classes. AUC = 1.0 is perfect,
            AUC = 0.5 is random guessing.
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for idx, f in enumerate(roc_files):
            with cols[idx % 2]:
                display_name = f.replace('_roc_curve.png', '').replace('_', ' ').title()
                st.markdown(f"**{display_name}**")
                st.image(os.path.join(RESULTS_DIR, f), use_container_width=True)

    # ---- Regression Section ----
    st.markdown("---")
    st.markdown(
        '<div class="section-header">Regression Model Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        <strong>Regression Task:</strong> Predict submission quality score
        (0-100) and final rank (position).<br>
        <strong>Metrics Used:</strong><br>
        - <strong>MAE</strong> (Mean Absolute Error) — Average size of prediction errors<br>
        - <strong>MSE</strong> (Mean Squared Error) — Penalizes large errors more heavily<br>
        - <strong>RMSE</strong> (Root MSE) — Error in the same units as the target<br>
        - <strong>R² Score</strong> — Proportion of variance explained
        (1.0 = perfect, 0.0 = same as predicting the mean, negative = worse than mean)
    </div>
    """, unsafe_allow_html=True)

    reg_comparison = os.path.join(RESULTS_DIR, 'regression_model_performance_comparison.png')
    if os.path.exists(reg_comparison):
        st.image(reg_comparison, use_container_width=True)

    # Actual vs Predicted plots
    avp_files = sorted([f for f in os.listdir(RESULTS_DIR) if 'actual_vs_predicted' in f])
    if avp_files:
        st.markdown(
            '<div class="section-header">Actual vs Predicted Plots</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="info-box">
            <strong>How to read these plots:</strong><br>
            Each point represents one test sample. The red dashed line shows
            where perfect predictions would fall. Points close to the line
            indicate accurate predictions. Widely scattered points indicate
            the model struggles to predict that target accurately.
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(2)
        for idx, f in enumerate(avp_files):
            with cols[idx % 2]:
                display_name = f.replace('_actual_vs_predicted.png', '').replace('_', ' ').replace(' - ', ': ').title()
                st.markdown(f"**{display_name}**")
                st.image(os.path.join(RESULTS_DIR, f), use_container_width=True)


# ===============================================
# PAGE 4: METHODOLOGY
# ===============================================
elif page == "Methodology":

    st.markdown(
        '<div class="main-title">Methodology</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Complete workflow and technical approach used in this project'
        '</div>',
        unsafe_allow_html=True
    )

    # Pipeline Overview
    st.markdown(
        '<div class="section-header">Project Pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-box">
        The project follows a standard machine learning pipeline with six stages.
        Each stage transforms the raw data progressively until we arrive at
        trained models capable of making predictions on new, unseen data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ```
    [Raw Dataset] → [Preprocessing] → [EDA] → [Train/Test Split] → [Model Training] → [Evaluation]
         |               |              |            |                    |                |
     303 rows      Encode + Scale   5 Plots      80:20 split       4 Algorithms      Metrics +
     10 columns    Save artifacts   Patterns     242 : 61          8 Models total     Plots
    ```
    """)

    # Preprocessing
    st.markdown(
        '<div class="section-header">1. Data Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    | Step | Operation | Method | Purpose |
    |------|-----------|--------|---------|
    | 1.1 | Categorical Encoding | LabelEncoder | Convert "Yes"/"No" text to 1/0 numbers for model input |
    | 1.2 | Feature Scaling | StandardScaler | Normalize all features to mean=0, std=1 so no single feature dominates |
    | 1.3 | Feature-Target Separation | Manual split | Separate input features (7) from output targets (3) |
    | 1.4 | Artifact Saving | Joblib | Save encoders and scaler as .pkl files for consistent inference |
    """)

    # Models
    st.markdown(
        '<div class="section-header">2. Models Implemented</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    | Algorithm | Type | How It Works | Strengths |
    |-----------|------|-------------|-----------|
    | **Passive Aggressive** | Online Learning | Updates model aggressively when it makes errors, passively when correct | Fast training, handles streaming data |
    | **Gaussian Process** | Probabilistic | Uses kernel functions (RBF) to model relationships as probability distributions | Provides uncertainty estimates, good for small data |
    | **Linear / Logistic** | Parametric | Fits a straight line (regression) or sigmoid curve (classification) to the data | Simple, interpretable, strong baseline |
    | **Stacking Ensemble** | Ensemble | Level 0: Random Forest + Gradient Boosting predict independently. Level 1: Their predictions are combined by a meta-model | Combines strengths of multiple algorithms |
    """)

    # Architecture diagram
    st.markdown(
        '<div class="section-header">3. Stacking Architecture (Best Model)</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ```
    Input Features (7)
         |
         ├──→ Random Forest (200 trees)    ──→ Predictions ──┐
         |                                                    ├──→ Meta-Model ──→ Final Prediction
         └──→ Gradient Boosting (200 trees) ──→ Predictions ──┘
                                                                    |
                                                           Linear Regression (regression)
                                                           Logistic Regression (classification)
    ```
    """)

    st.markdown("""
    <div class="info-box">
        <strong>Why Stacking?</strong><br>
        Individual models have different strengths and weaknesses.
        Random Forest handles non-linear patterns well but may overfit.
        Gradient Boosting corrects errors iteratively but is sensitive to noise.
        By stacking them, the meta-model learns which base model to trust
        in different situations, often producing better overall predictions.
    </div>
    """, unsafe_allow_html=True)

    # Evaluation
    st.markdown(
        '<div class="section-header">4. Evaluation Approach</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    | Metric | Task | What It Measures |
    |--------|------|-----------------|
    | Accuracy | Classification | Overall percentage of correct predictions |
    | Precision | Classification | Correctness of positive predictions |
    | Recall | Classification | Coverage of actual positive cases |
    | F1-Score | Classification | Balance between Precision and Recall |
    | MAE | Regression | Average magnitude of errors (in original units) |
    | MSE | Regression | Average of squared errors (penalizes large errors) |
    | RMSE | Regression | Square root of MSE (interpretable error magnitude) |
    | R² Score | Regression | Proportion of variance explained by the model |
    """)

    # Challenges
    st.markdown(
        '<div class="section-header">5. Challenges Identified</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    | Challenge | Impact | Proposed Solution |
    |-----------|--------|------------------|
    | Class Imbalance | 85% teams are non-winners; models biased toward predicting "No" | SMOTE oversampling, class weights, threshold tuning |
    | Small Dataset | Only 303 samples limits model learning capacity | Collect more hackathon data from multiple events |
    | Limited Features | Current 7 features may not capture all performance factors | Add features like GitHub activity, tech stack, domain |
    | Negative R² Scores | Models perform worse than simple mean prediction | Feature engineering, polynomial features, regularization |
    """)


# ===============================================
# PAGE 5: ABOUT
# ===============================================
elif page == "About":

    st.markdown(
        '<div class="main-title">About This Project</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="main-subtitle">'
        'Project details, technology stack, and team information'
        '</div>',
        unsafe_allow_html=True
    )

    # Project Overview
    st.markdown(
        '<div class="section-header">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This project develops a machine learning system for predicting
    hackathon team performance based on measurable team dynamics.
    Given seven input characteristics of a hackathon team, the system
    predicts three performance outcomes using trained ML models.
    """)

    st.markdown("""
    | Input Features | Output Predictions |
    |---------------|-------------------|
    | Team Size | Submission Quality Score (0-100) |
    | Skill Diversity Score | Final Rank (position) |
    | Average Experience Level | Won Hackathon (Yes / No) |
    | Conflict Count | |
    | Communication Rating | |
    | Hours Spent | |
    | Mentor Guidance (Yes/No) | |
    """)

    # Tech Stack
    st.markdown(
        '<div class="section-header">Technology Stack</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    | Component | Technology | Version | Purpose |
    |-----------|-----------|---------|---------|
    | Programming Language | Python | 3.11.4 | Core development |
    | ML Framework | Scikit-learn | 1.6.1 | Model training and evaluation |
    | Data Processing | Pandas, NumPy | Latest | Data manipulation and computation |
    | Visualization | Matplotlib, Seaborn | Latest | Charts, plots, and graphs |
    | Web Application | Streamlit | 1.54.0 | Interactive user interface |
    | Model Persistence | Joblib | Latest | Saving and loading trained models |
    | IDE | Visual Studio Code | Latest | Development environment |
    """)

    # Project Structure
    st.markdown(
        '<div class="section-header">Project Structure</div>',
        unsafe_allow_html=True
    )

    st.code("""
    HackathonProject/
    ├── Dataset/
    │   ├── Hackathon_Team_Dynamics_Dataset.csv   (Training data)
    │   └── testdata.csv                          (Test data)
    ├── models/                                   (Saved trained models)
    │   ├── stacking_rf_gb_regressor.pkl
    │   ├── stacking_rf_gb_classifier.pkl
    │   ├── standard_scaler.pkl
    │   └── ... (11 .pkl files total)
    ├── results/                                  (Evaluation plots)
    │   ├── confusion matrices
    │   ├── ROC curves
    │   └── actual vs predicted plots
    ├── train.py                                  (Training pipeline)
    ├── app.py                                    (Web application)
    └── requirements.txt                          (Dependencies)
    """, language="text")

    # Team
    st.markdown(
        '<div class="section-header">Team Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    **Batch:** 9

    **Project Title:** Hackathon Team Dynamics — Performance Prediction System

    **Project Type:** Mini Project (Machine Learning)
    """)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
        "Developed as part of college mini project review"
        "</div>",
        unsafe_allow_html=True
    )