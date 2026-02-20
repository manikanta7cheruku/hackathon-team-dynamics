# =============================================================================
# HACKATHON TEAM DYNAMICS - COMPLETE TRAINING PIPELINE
# =============================================================================
# This script does everything:
# 1. Loads the dataset
# 2. Preprocesses (encode + scale)
# 3. Performs EDA (Exploratory Data Analysis)
# 4. Splits data into train/test
# 5. Trains 4 regression models + 4 classification models
# 6. Evaluates all models with metrics + plots
# 7. Compares all models
# 8. Saves everything (models, plots, encoders, scaler)
# =============================================================================

# -----------------------------------------------
# IMPORTS
# -----------------------------------------------
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

# Visualization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (saves plots without showing)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

# Train-Test Split
from sklearn.model_selection import train_test_split

# Models - Regression
from sklearn.linear_model import LinearRegression, PassiveAggressiveRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)

# Models - Classification
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
)

# Multi-output wrapper (for predicting 2 regression targets at once)
from sklearn.multioutput import MultiOutputRegressor

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, auc
)

# Save/Load models
import joblib

# Suppress warnings for clean output
warnings.filterwarnings('ignore')


# -----------------------------------------------
# CONFIGURATION - Directories
# -----------------------------------------------
MODEL_DIR = "models"        # Where trained models are saved
RESULTS_DIR = "results"     # Where plots/images are saved
DATASET_PATH = "Dataset/Hackathon_Team_Dynamics_Dataset.csv"

# Create directories if they don't exist
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Global DataFrames to store metrics for comparison
classification_metrics_df = pd.DataFrame(
    columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
)
regression_metrics_df = pd.DataFrame(
    columns=['Algorithm', 'MAE', 'MSE', 'RMSE', 'R2']
)


# =============================================================================
# STEP 1: LOAD DATASET
# =============================================================================
def load_dataset(file_path):
    """
    Load the hackathon dataset from a CSV file.
    Returns a pandas DataFrame or None if file not found.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[✓] Dataset loaded successfully | Shape: {df.shape}")
        print(f"[✓] Columns: {list(df.columns)}")
        print(f"[✓] First 5 rows:")
        print(df.head())
        print()
        return df
    except FileNotFoundError:
        print(f"[✗] Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"[✗] Error loading dataset: {e}")
        return None


# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================
def preprocess_data(df, is_train=True):
    """
    Preprocess the hackathon dataset:
    - Encode categorical columns (mentor_guidance, won_hackathon)
    - Standard scale all numeric features
    - Separate features (X) from targets (y_reg, y_class)
    
    Parameters:
        df       : Raw DataFrame
        is_train : If True, fit encoders/scaler and save them
                   If False, load saved encoders/scaler and transform
    
    Returns:
        X       : Feature matrix (7 columns, scaled)
        y_reg   : Regression targets (submission_quality_score, final_rank)
        y_class : Classification target (won_hackathon: 0 or 1)
    """
    df = df.copy()

    # ---- Encode 'mentor_guidance' (Yes/No → 1/0) ----
    feature_encoder = LabelEncoder()
    df['mentor_guidance'] = feature_encoder.fit_transform(
        df['mentor_guidance'].astype(str)
    )

    # ---- Encode 'won_hackathon' (Yes/No → 1/0) ----
    target_encoder = LabelEncoder()
    df['won_hackathon'] = target_encoder.fit_transform(
        df['won_hackathon'].astype(str)
    )

    # ---- Separate features from targets ----
    # X = everything EXCEPT the 3 target columns
    X = df.drop(
        columns=['submission_quality_score', 'final_rank', 'won_hackathon']
    )

    # ---- Standard Scaling on numeric features ----
    # StandardScaler: transforms data to mean=0, std=1
    # This helps models that are sensitive to feature scale
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

    if is_train:
        # Fit scaler on training data and save it
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        joblib.dump(scaler, scaler_path)
        print("[✓] StandardScaler fitted and saved")
    else:
        # Load saved scaler and just transform (don't fit again)
        scaler = joblib.load(scaler_path)
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        print("[✓] StandardScaler loaded and applied")

    # ---- Define targets ----
    # Regression: predict these 2 numeric values
    y_reg = df[['submission_quality_score', 'final_rank']]

    # Classification: predict this binary value (0 or 1)
    y_class = df['won_hackathon']

    # ---- Save encoders for later use (prediction on new data) ----
    if is_train:
        joblib.dump(
            feature_encoder,
            os.path.join(MODEL_DIR, 'mentor_guidance_encoder.pkl')
        )
        joblib.dump(
            target_encoder,
            os.path.join(MODEL_DIR, 'won_hackathon_encoder.pkl')
        )
        print("[✓] Encoders saved")

    print(f"[✓] Features shape: {X.shape}")
    print(f"[✓] Regression targets shape: {y_reg.shape}")
    print(f"[✓] Classification target shape: {y_class.shape}")
    print()

    return X, y_reg, y_class


# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
def perform_eda(df):
    """
    Generate 5 EDA plots and save as 'eda_plots.png':
    1. Distribution of submission quality score
    2. Hours spent vs submission quality (colored by win)
    3. Mentor guidance vs winning
    4. Team size vs final rank
    5. Count of won_hackathon (class balance)
    """
    print("[...] Generating EDA plots...")

    plt.figure(figsize=(15, 10))

    # Plot 1: How are submission scores distributed?
    plt.subplot(2, 3, 1)
    sns.histplot(df['submission_quality_score'], kde=True)
    plt.title('Distribution of Submission Quality Score')

    # Plot 2: Do more hours = better quality? Does winning matter?
    plt.subplot(2, 3, 2)
    sns.scatterplot(
        x='hours_spent',
        y='submission_quality_score',
        hue='won_hackathon',
        data=df
    )
    plt.title('Hours Spent vs Submission Quality')

    # Plot 3: Does having a mentor help win?
    plt.subplot(2, 3, 3)
    sns.countplot(x='mentor_guidance', hue='won_hackathon', data=df)
    plt.title('Mentor Guidance vs Winning')

    # Plot 4: Does team size affect final rank?
    plt.subplot(2, 3, 4)
    sns.boxplot(x='team_size', y='final_rank', data=df)
    plt.title('Final Rank by Team Size')

    # Plot 5: How many teams won vs lost? (class balance check)
    plt.subplot(2, 3, 5)
    sns.countplot(x='won_hackathon', data=df)
    plt.title('Count Plot of Won Hackathon')

    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("[✓] EDA plots saved as 'eda_plots.png'")
    print()


# =============================================================================
# STEP 4: TRAIN-TEST SPLIT
# =============================================================================
def split_data(X, y_reg, y_class, test_size=0.2, random_state=42):
    """
    Split data into 80% training and 20% testing.
    
    We split separately for regression targets and classification target,
    but using the SAME random_state ensures the same rows go to train/test.
    
    Returns:
        X_train, X_test           : Feature matrices
        y_reg_train, y_reg_test   : Regression targets
        y_class_train, y_class_test : Classification target
    """
    # Split features + regression targets
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )

    # Split features + classification target (same split)
    _, _, y_class_train, y_class_test = train_test_split(
        X, y_class, test_size=test_size, random_state=random_state
    )

    print(f"[✓] Train set: {X_train.shape[0]} samples")
    print(f"[✓] Test set:  {X_test.shape[0]} samples")
    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"    y_reg_train: {y_reg_train.shape}, y_reg_test: {y_reg_test.shape}")
    print(f"    y_class_train: {y_class_train.shape}, y_class_test: {y_class_test.shape}")
    print()

    return X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test


# =============================================================================
# STEP 5: EVALUATION METRICS
# =============================================================================
def calculate_metrics(algorithm, y_pred, y_test, task_type='classification', y_score=None):
    """
    Calculate and display evaluation metrics.
    Also generates and saves plots (confusion matrix, ROC curve, scatter plots).
    
    Parameters:
        algorithm : Name of the model (used for labeling)
        y_pred    : Predicted values
        y_test    : Actual values
        task_type : 'classification' or 'regression'
        y_score   : Probability scores for ROC curve (classification only)
    """
    global classification_metrics_df, regression_metrics_df

    if task_type == 'classification':
        # ---- Load target encoder to get class names ----
        try:
            le_target = joblib.load(
                os.path.join(MODEL_DIR, 'won_hackathon_encoder.pkl')
            )
            categories = le_target.classes_  # ['No', 'Yes']
        except:
            categories = np.unique(y_test)

        # ---- Calculate classification metrics ----
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

        # Store in global DataFrame for later comparison
        classification_metrics_df.loc[len(classification_metrics_df)] = [
            algorithm, acc, prec, rec, f1
        ]

        # Print results
        print(f"  {algorithm} | Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | "
              f"Recall: {rec:.2f}% | F1: {f1:.2f}%")
        print(f"  Classification Report:")
        print(classification_report(
            y_test, y_pred,
            labels=range(len(categories)),
            target_names=categories,
            zero_division=0
        ))

        # ---- Confusion Matrix Plot ----
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True,
            xticklabels=categories,
            yticklabels=categories,
            cmap="Blues", fmt="g"
        )
        plt.title(f'{algorithm} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        filename = f"{algorithm.replace(' ', '_').lower()}_confusion_matrix.png"
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
        plt.close()

        # ---- ROC Curve Plot (only if probability scores available) ----
        if y_score is not None:
            plt.figure(figsize=(10, 8))
            if len(categories) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr,
                    label=f'{categories[1]} vs {categories[0]} (AUC = {roc_auc:.2f})'
                )
            plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
            plt.title(f"{algorithm} - ROC Curve")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            filename = f"{algorithm.replace(' ', '_').lower()}_roc_curve.png"
            plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
            plt.close()

    elif task_type == 'regression':
        # ---- Calculate regression metrics ----
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Store in global DataFrame for later comparison
        regression_metrics_df.loc[len(regression_metrics_df)] = [
            algorithm, mae, mse, rmse, r2
        ]

        # Print results
        print(f"  {algorithm} | MAE: {mae:.4f} | MSE: {mse:.4f} | "
              f"RMSE: {rmse:.4f} | R²: {r2:.4f}")

        # ---- Actual vs Predicted Scatter Plot ----
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        # Red dashed line = perfect prediction line
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', label='Perfect Prediction'
        )
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{algorithm} - Actual vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"{algorithm.replace(' ', '_').lower()}_actual_vs_predicted.png"
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
        plt.close()


# =============================================================================
# STEP 6: MODEL TRAINING FUNCTIONS
# =============================================================================

# -----------------------------------------------
# MODEL 1: Passive Aggressive (Regression + Classification)
# -----------------------------------------------
# What: Online learning algorithm that updates aggressively on misclassified samples
# Why:  Fast, works well with streaming/large data, simple baseline
# -----------------------------------------------
def train_passive_aggressive(X_train, y_train, X_test, y_test, model_name, task_type='regression'):
    """Train Passive Aggressive model for regression or classification."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if os.path.exists(model_path):
        print(f"  [↻] Loading saved {model_name}...")
        model = joblib.load(model_path)
    else:
        print(f"  [⚙] Training {model_name}...")

        if task_type == 'regression':
            base = PassiveAggressiveRegressor(
                max_iter=1000, random_state=42, tol=1e-3
            )
            # Wrap in MultiOutputRegressor if predicting 2 targets
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
                model = MultiOutputRegressor(base)
            else:
                model = base
        else:
            model = PassiveAggressiveClassifier(
                max_iter=1000, random_state=42, tol=1e-3
            )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"  [✓] Model saved to {model_path}")

    # Predict and evaluate
    y_pred = model.predict(X_test)
    _evaluate_model(model_name, y_train, y_pred, y_test, task_type, model)
    return model


# -----------------------------------------------
# MODEL 2: Gaussian Process (Regression + Classification)
# -----------------------------------------------
# What: Probabilistic model using kernel functions (RBF kernel here)
# Why:  Gives uncertainty estimates, works well on small datasets
# -----------------------------------------------
def train_gaussian_process(X_train, y_train, X_test, y_test, model_name, task_type='regression'):
    """Train Gaussian Process model for regression or classification."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if os.path.exists(model_path):
        print(f"  [↻] Loading saved {model_name}...")
        model = joblib.load(model_path)
    else:
        print(f"  [⚙] Training {model_name}...")

        # RBF kernel: measures similarity between data points
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
        )

        if task_type == 'regression':
            base = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-2,
                normalize_y=True, random_state=42
            )
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
                model = MultiOutputRegressor(base)
            else:
                model = base
        else:
            model = GaussianProcessClassifier(
                kernel=kernel, random_state=42
            )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"  [✓] Model saved to {model_path}")

    y_pred = model.predict(X_test)
    _evaluate_model(model_name, y_train, y_pred, y_test, task_type, model)
    return model


# -----------------------------------------------
# MODEL 3: Linear Regression + Logistic Regression
# -----------------------------------------------
# What: Linear Regression = straight line fit for numeric prediction
#        Logistic Regression = sigmoid curve for probability/classification
# Why:  Simple, interpretable, good baseline to compare against complex models
# -----------------------------------------------
def train_linear_logistic(X_train, y_train, X_test, y_test, model_name, task_type='regression'):
    """Train Linear or Logistic Regression model."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if os.path.exists(model_path):
        print(f"  [↻] Loading saved {model_name}...")
        model = joblib.load(model_path)
    else:
        print(f"  [⚙] Training {model_name}...")

        if task_type == 'regression':
            base = LinearRegression()
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
                model = MultiOutputRegressor(base)
            else:
                model = base
        else:
            model = LogisticRegression(
                max_iter=1000, solver='lbfgs', random_state=42
            )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"  [✓] Model saved to {model_path}")

    y_pred = model.predict(X_test)
    _evaluate_model(model_name, y_train, y_pred, y_test, task_type, model)
    return model


# -----------------------------------------------
# MODEL 4: Stacking (Random Forest + Gradient Boosting)
# -----------------------------------------------
# What: Ensemble method - combines predictions of RF and GB
#        Level 0: RF and GB make predictions independently
#        Level 1: Linear/Logistic Regression combines those predictions
# Why:  Most powerful approach, reduces individual model weaknesses
# -----------------------------------------------
def train_stacking(X_train, y_train, X_test, y_test, model_name, task_type='regression'):
    """Train Stacking Ensemble model (RF + GB as base, Linear/Logistic as meta)."""
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if os.path.exists(model_path):
        print(f"  [↻] Loading saved {model_name}...")
        model = joblib.load(model_path)
    else:
        print(f"  [⚙] Training {model_name}... (this may take a moment)")

        if task_type == 'regression':
            # Base models (Level 0)
            base_estimators = [
                ('rf', RandomForestRegressor(
                    n_estimators=200, random_state=42, n_jobs=-1
                )),
                ('gbr', GradientBoostingRegressor(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=3, random_state=42
                ))
            ]
            # Meta model (Level 1) - combines base predictions
            meta = LinearRegression()

            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
                stacking = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=meta,
                    passthrough=True, n_jobs=-1
                )
                model = MultiOutputRegressor(stacking)
            else:
                model = StackingRegressor(
                    estimators=base_estimators,
                    final_estimator=meta,
                    passthrough=True, n_jobs=-1
                )
        else:
            base_estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=-1,
                    class_weight='balanced'
                )),
                ('gbc', GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05,
                    max_depth=3, random_state=42
                ))
            ]
            meta = LogisticRegression(max_iter=1000, class_weight='balanced')

            model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta,
                passthrough=True, n_jobs=-1
            )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"  [✓] Model saved to {model_path}")

    y_pred = model.predict(X_test)
    _evaluate_model(model_name, y_train, y_pred, y_test, task_type, model)
    return model


# -----------------------------------------------
# HELPER: Evaluate any model (used by all training functions)
# -----------------------------------------------
def _evaluate_model(model_name, y_train, y_pred, y_test, task_type, model):
    """Route evaluation to correct metric calculator."""
    if task_type == 'regression':
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            for i, col in enumerate(y_train.columns):
                calculate_metrics(
                    f"{model_name} - {col}",
                    y_pred[:, i],
                    y_test.iloc[:, i],
                    task_type='regression'
                )
        else:
            calculate_metrics(model_name, y_pred, y_test, task_type='regression')
    else:
        # Get probability scores for ROC curve (if model supports it)
        y_score = None
        if hasattr(model, 'predict_proba'):
            try:
                y_score = model.predict_proba(
                    y_test.values.reshape(-1, 1) if hasattr(y_test, 'values')
                    else y_test
                )
            except:
                # Some models need X_test, not y_test for predict_proba
                # We handle this in the calling function
                y_score = None

        calculate_metrics(
            model_name, y_pred, y_test,
            task_type='classification', y_score=y_score
        )


# =============================================================================
# STEP 7: MODEL COMPARISON PLOTS
# =============================================================================
def plot_model_comparison():
    """Generate comparison bar charts for all trained models."""
    global classification_metrics_df, regression_metrics_df

    # ---- Classification Comparison ----
    if not classification_metrics_df.empty:
        print("\n[...] Generating classification comparison plot...")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

        for idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1-Score']):
            axes[idx].barh(
                classification_metrics_df['Algorithm'],
                classification_metrics_df[metric],
                color=colors[idx]
            )
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].set_xlim(0, 105)

        plt.suptitle(
            'Classification Model Performance Comparison',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, 'model_performance_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("[✓] Classification comparison saved")

    # ---- Regression Comparison ----
    if not regression_metrics_df.empty:
        print("[...] Generating regression comparison plot...")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['#FF5722', '#9C27B0', '#00BCD4', '#8BC34A']

        for idx, metric in enumerate(['MAE', 'MSE', 'RMSE', 'R2']):
            axes[idx].barh(
                regression_metrics_df['Algorithm'],
                regression_metrics_df[metric],
                color=colors[idx]
            )
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')

        plt.suptitle(
            'Regression Model Performance Comparison',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, 'regression_model_performance_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        print("[✓] Regression comparison saved")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    print("=" * 60)
    print(" HACKATHON TEAM DYNAMICS - TRAINING PIPELINE")
    print("=" * 60)
    print()

    # ---- Step 1: Load Data ----
    print("─" * 40)
    print("STEP 1: Loading Dataset")
    print("─" * 40)
    df = load_dataset(DATASET_PATH)
    if df is None:
        exit()

    # ---- Step 2: EDA ----
    print("─" * 40)
    print("STEP 2: Exploratory Data Analysis")
    print("─" * 40)
    perform_eda(df)

    # ---- Step 3: Preprocessing ----
    print("─" * 40)
    print("STEP 3: Preprocessing")
    print("─" * 40)
    X, y_reg, y_class = preprocess_data(df, is_train=True)

    # ---- Step 4: Train-Test Split ----
    print("─" * 40)
    print("STEP 4: Train-Test Split")
    print("─" * 40)
    X_train, X_test, yreg_train, yreg_test, yclass_train, yclass_test = \
        split_data(X, y_reg, y_class)

    # ---- Step 5: Train All Models ----

    # --- Model 1: Passive Aggressive ---
    print("─" * 40)
    print("MODEL 1: Passive Aggressive")
    print("─" * 40)
    train_passive_aggressive(
        X_train, yreg_train, X_test, yreg_test,
        model_name="passive_aggressive_regressor", task_type='regression'
    )
    train_passive_aggressive(
        X_train, yclass_train, X_test, yclass_test,
        model_name="passive_aggressive_classifier", task_type='classification'
    )

    # --- Model 2: Gaussian Process ---
    print("─" * 40)
    print("MODEL 2: Gaussian Process")
    print("─" * 40)
    train_gaussian_process(
        X_train, yreg_train, X_test, yreg_test,
        model_name="gaussian_process_regressor", task_type='regression'
    )
    train_gaussian_process(
        X_train, yclass_train, X_test, yclass_test,
        model_name="gaussian_process_classifier", task_type='classification'
    )

    # --- Model 3: Linear / Logistic ---
    print("─" * 40)
    print("MODEL 3: Linear / Logistic Regression")
    print("─" * 40)
    train_linear_logistic(
        X_train, yreg_train, X_test, yreg_test,
        model_name="linear_regressor", task_type='regression'
    )
    train_linear_logistic(
        X_train, yclass_train, X_test, yclass_test,
        model_name="logistic_model", task_type='classification'
    )

    # --- Model 4: Stacking (RF + GB) ---
    print("─" * 40)
    print("MODEL 4: Stacking (Random Forest + Gradient Boosting)")
    print("─" * 40)
    train_stacking(
        X_train, yreg_train, X_test, yreg_test,
        model_name="stacking_rf_gb_regressor", task_type='regression'
    )
    train_stacking(
        X_train, yclass_train, X_test, yclass_test,
        model_name="stacking_rf_gb_classifier", task_type='classification'
    )

    # ---- Step 6: Model Comparison ----
    print()
    print("─" * 40)
    print("STEP 6: Model Comparison")
    print("─" * 40)
    plot_model_comparison()

    # ---- Print Final Summary Tables ----
    print()
    print("=" * 60)
    print(" FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\n📊 Classification Results:")
    print(classification_metrics_df.to_string(index=False))

    print("\n📈 Regression Results:")
    print(regression_metrics_df.to_string(index=False))

    print()
    print("=" * 60)
    print(" ✅ TRAINING COMPLETE!")
    print(f" Models saved in: ./{MODEL_DIR}/")
    print(f" Plots saved in:  ./{RESULTS_DIR}/")
    print(f" EDA saved as:    ./eda_plots.png")
    print("=" * 60)