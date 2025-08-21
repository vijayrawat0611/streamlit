# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

st.title("Breast Cancer Classification - Model Comparison")

uploaded_file = st.file_uploader("Upload your Breast Cancer CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Identify target column
    possible_targets = ['diagnosis', 'target', 'class', 'label', 'y', 'A Stage']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        for c in df.columns:
            if df[c].nunique() == 2:
                target_col = c
                break
    if target_col is None:
        target_col = df.columns[-1]

    st.write(f"**Detected Target Column:** {target_col}")

    # Preprocess
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical target
    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Encode categorical predictors
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def get_metrics(y_true, y_pred, y_proba):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    st.subheader("Training Deep Learning Model (Keras)")

    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    with st.spinner("Training deep learning model..."):
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )

    y_pred_proba_dl = model.predict(X_test_scaled).ravel()
    y_pred_dl = (y_pred_proba_dl >= 0.5).astype(int)
    dl_results = {"model": "DeepLearning_Keras", **get_metrics(y_test, y_pred_dl, y_pred_proba_dl)}

    st.write("### Deep Learning Model Performance")
    st.write(f"**Accuracy:** {dl_results['accuracy']:.4f}")
    st.write(f"**Precision:** {dl_results['precision']:.4f}")
    st.write(f"**Recall:** {dl_results['recall']:.4f}")
    st.write(f"**F1-score:** {dl_results['f1']:.4f}")
    st.write(f"**ROC AUC:** {dl_results['roc_auc']:.4f}")
    st.write("Confusion Matrix:")
    st.write(pd.DataFrame(dl_results["confusion_matrix"], index=["Actual Neg", "Actual Pos"], columns=["Predicted Neg", "Predicted Pos"]))

    # Plot training curves
    st.subheader("Training and Validation Accuracy Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    st.subheader("Training and Validation Loss Curve")
    fig1b, ax1b = plt.subplots()
    ax1b.plot(history.history['loss'], label='Train Loss')
    ax1b.plot(history.history['val_loss'], label='Val Loss')
    ax1b.set_xlabel('Epoch')
    ax1b.set_ylabel('Loss')
    ax1b.legend()
    ax1b.grid(True)
    st.pyplot(fig1b)

    # Traditional ML models
    st.subheader("Training Traditional Machine Learning Models")

    ml_models = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    ml_results = []
    progress_bar = st.progress(0)
    for i, (name, clf) in enumerate(ml_models.items()):
        st.write(f"Training {name}...")
        if name in ["LogisticRegression", "SVM"]:
            clf.fit(X_train_scaled, y_train)
            y_proba = clf.predict_proba(X_test_scaled)[:, 1]
            y_pred = clf.predict(X_test_scaled)
        else:
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

        ml_results.append({"model": name, **get_metrics(y_test, y_pred, y_proba)})
        progress_bar.progress((i + 1) / len(ml_models))

    # Combine results
    all_results = [dl_results] + ml_results

    # Save results CSV
    csv_path = "breast_cancer_model_comparison_results.csv"
    if os.path.exists(csv_path):
        df_results = pd.read_csv(csv_path)
        df_results = pd.concat([df_results, pd.DataFrame(all_results)], ignore_index=True)
    else:
        df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_path, index=False)
    st.success(f"Results saved to {csv_path}")

    # Show results
    st.subheader("Model Performance Comparison Table")
    st.dataframe(df_results.style.format({
        "accuracy": "{:.4f}",
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1": "{:.4f}",
        "roc_auc": "{:.4f}"
    }))

    # Bar chart for comparison
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df_plot = pd.DataFrame(all_results)

    st.subheader("Model Metrics Comparison - Bar Chart")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    width = 0.12
    x = np.arange(len(df_plot["model"]))
    for i, metric in enumerate(metrics_to_plot):
        ax2.bar(x + i * width, df_plot[metric], width=width, label=metric)

    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(df_plot["model"], rotation=45, ha='right')
    ax2.set_ylabel("Score")
    ax2.set_title("Model Metrics Comparison")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to get started.")
