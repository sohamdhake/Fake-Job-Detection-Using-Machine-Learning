"""
Fake Job Detection System - Model Training Script
=================================================
This script:
1. Loads the Real/Fake Job Postings dataset (or downloads it if missing).
2. Cleans and preprocesses the text data (Title, Company Profile, Description, Requirements).
3. Converts text into numerical features using TF-IDF Vectorization.
4. Trains a Logistic Regression model with balanced class weighting.
5. Evaluates model performance using a stratified train/test split.
6. Displays actual Accuracy, Precision, Recall, and F1-score.
7. Saves the trained model, TF-IDF vectorizer, and evaluation metrics into the 'models/' folder.
"""

import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH_DEFAULT = os.path.join(BASE_DIR, "fake_job_postings_sample.csv")
CSV_PATH_DATA = os.path.join(DATA_DIR, "fake_job_postings_sample.csv")

# Public dataset source URL (Raw Kaggle EMSCAD mirror on GitHub)
DATASET_URL = (
    "https://raw.githubusercontent.com/Cindyalifia/bangkit-project-1/master/fake_job_postings_sample.csv"
)
KAGGLE_SOURCE_URL = "https://www.kaggle.com/datasets/shivamb/real-or-fake-job-posting-prediction"


def clean_text(text: str) -> str:
    """
    Cleans raw text input:
    - Lowercases text
    - Removes URLs and email addresses
    - Removes special symbols and non-alphanumeric noise
    - Collapses multiple whitespace into a single space
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove special characters / punctuation keep words and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_dataset():
    """
    Checks if the dataset exists locally.
    If not, attempts automatic download from trusted repository,
    or gives step-by-step instructions for Kaggle.
    """
    if os.path.exists(CSV_PATH_DEFAULT):
        return CSV_PATH_DEFAULT
    if os.path.exists(CSV_PATH_DATA):
        return CSV_PATH_DATA

    print("=" * 65)
    print("DATASET NOT FOUND LOCALLY")
    print("=" * 65)
    print(f"Target location: {CSV_PATH_DEFAULT}")
    print("Attempting automatic download from public verified source...")
    print(f"Source URL: {DATASET_URL}")

    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(
                    f"\rDownloading dataset: {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(DATASET_URL, CSV_PATH_DEFAULT, reporthook=reporthook)
        print("\n[OK] Dataset downloaded successfully!\n")
        return CSV_PATH_DEFAULT
    except Exception as e:
        print(f"\n[ERROR] Automatic download failed: {e}")
        print("\n" + "=" * 65)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 65)
        print("1. Visit Kaggle's Real / Fake Job Posting Prediction dataset:")
        print(f"   {KAGGLE_SOURCE_URL}")
        print("2. Download 'fake_job_postings_sample.csv' (or unzip archive.zip).")
        print(f"3. Place the 'fake_job_postings_sample.csv' file inside this folder:")
        print(f"   {BASE_DIR}")
        print("4. Re-run: py train.py")
        print("=" * 65 + "\n")
        raise FileNotFoundError(
            f"Please download fake_job_postings_sample.csv and place it at {CSV_PATH_DEFAULT}"
        )


def main():
    print("=" * 65)
    print("       FAKE JOB DETECTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 65)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Acquire dataset
    dataset_path = ensure_dataset()

    print(f"Loading dataset from: {dataset_path}...")
    df = pd.read_csv(dataset_path)

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Verify target column exists
    if "fraudulent" not in df.columns:
        raise ValueError("Column 'fraudulent' not found in dataset!")

    # Check class distribution
    class_counts = df["fraudulent"].value_counts().to_dict()
    real_count = int(class_counts.get(0, 0))
    fake_count = int(class_counts.get(1, 0))
    print(f"Class Distribution: {real_count} Real Jobs (0) | {fake_count} Fake Jobs (1)")
    print(f"Fraudulent percentage: {(fake_count / len(df)) * 100:.2f}%\n")

    # 2. Feature Combination
    print("Step 1: Combining text fields (Title, Company Profile, Description, Requirements)...")
    # Fill missing values with empty string
    title = df["title"].fillna("") if "title" in df.columns else ""
    company_profile = (
        df["company_profile"].fillna("") if "company_profile" in df.columns else ""
    )
    description = df["description"].fillna("") if "description" in df.columns else ""
    requirements = df["requirements"].fillna("") if "requirements" in df.columns else ""
    benefits = df["benefits"].fillna("") if "benefits" in df.columns else ""

    # Combine all textual information
    combined = (
        title
        + " "
        + company_profile
        + " "
        + description
        + " "
        + requirements
        + " "
        + benefits
    )

    print("Step 2: Cleaning and preprocessing text (lowercasing, token cleansing)...")
    cleaned_texts = combined.apply(clean_text)

    X = cleaned_texts
    y = df["fraudulent"].astype(int)

    # 3. Stratified Train/Test Split
    print("Step 3: Performing Stratified Train-Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} samples ({sum(y_train == 1)} fake)")
    print(f"Testing set:  {len(X_test)} samples ({sum(y_test == 1)} fake)\n")

    # 4. TF-IDF Vectorization
    print("Step 4: Extracting TF-IDF Features (ngram_range=(1,2), max_features=10000)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")

    # 5. Logistic Regression Model Training
    print("Step 5: Training Logistic Regression Model (class_weight='balanced')...")
    # class_weight='balanced' ensures the model appropriately detects the minority fake job class
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X_train_tfidf, y_train)
    print("[OK] Model training complete!\n")

    # 6. Evaluation
    print("Step 6: Evaluating Model on Test Data...")
    y_pred = model.predict(X_test_tfidf)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("=" * 65)
    print("             ACTUAL MODEL EVALUATION METRICS")
    print("=" * 65)
    print(f"  Accuracy:  {acc * 100:.2f}%  (Overall correct classifications)")
    print(f"  Precision: {prec * 100:.2f}%  (When model flags fake, how often it's right)")
    print(f"  Recall:    {rec * 100:.2f}%  (How many actual fake jobs were caught)")
    print(f"  F1-Score:  {f1 * 100:.2f}%  (Harmonic mean of precision & recall)")
    print("=" * 65)
    print("\nConfusion Matrix:")
    print(f"  [TN={cm[0][0]:<5}  FP={cm[0][1]:<5}]  (Real Jobs)")
    print(f"  [FN={cm[1][0]:<5}  TP={cm[1][1]:<5}]  (Fake Jobs)\n")

    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real Job (0)", "Fake Job (1)"]))

    # 7. Save Artifacts
    model_path = os.path.join(MODELS_DIR, "model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")

    print("Step 7: Saving trained model, vectorizer, and metrics...")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    metrics_data = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "real_postings": real_count,
        "fake_postings": fake_count,
        "model_name": "Logistic Regression (TF-IDF)",
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[OK] Model saved to:      {model_path}")
    print(f"[OK] Vectorizer saved to: {vectorizer_path}")
    print(f"[OK] Metrics saved to:    {metrics_path}\n")

    # 8. Sanity check predictions
    print("Step 8: Running Sanity Predictions on Sample Test Cases...")
    sample_tests = [
        {
            "label": "Expected Fake",
            "text": clean_text(
                "Urgent Data Entry Clerks Needed. Earn $500 to $800 daily working 2 hours from home. "
                "No experience or degree required! Instant hiring without interview. Send registration fee "
                "via gift cards or CashApp to secure your starter kit and laptop."
            ),
        },
        {
            "label": "Expected Real",
            "text": clean_text(
                "Senior Python Backend Engineer at CloudTech Systems. Seeking experienced developer with 4+ years "
                "in Python, FastAPI, PostgreSQL, and Docker. Responsibilities include building scalable microservices, "
                "writing unit tests, and collaborating with cross-functional agile teams. Competitive salary, health "
                "insurance, 401(k) matching, and paid time off included."
            ),
        },
    ]

    for test in sample_tests:
        vec = vectorizer.transform([test["text"]])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        verdict = "FAKE / FRAUDULENT (1)" if pred == 1 else "REAL / LEGITIMATE (0)"
        confidence = prob[pred] * 100
        print(f"  Test [{test['label']}]: -> Verdict: {verdict} ({confidence:.1f}% confidence)")

    print("\n" + "=" * 65)
    print("SUCCESS: Training completed successfully!")
    print("You can now launch the web interface with: streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
