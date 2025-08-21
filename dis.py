# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

st.title("Breast Cancer Classification - Deep Learning with Keras")
st.write("Upload your CSV file and train a deep learning model.")

# ====== 1. File uploader ======
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

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

    # ====== 2. Preprocess ======
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical target
    if y.dtype == object or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y)

    # Encode categorical predictors
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median())

    # ====== 3. Train/test split ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ====== 4. Scale features ======
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ====== 5. Build deep learning model ======
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # ====== 6. Train ======
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    with st.spinner("Training model..."):
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=32,
            callbacks=[es],
            verbose=0
        )

    # ====== 7. Predictions ======
    y_pred_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # ====== 8. Metrics ======
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Display metrics
    st.subheader("📊 Model Performance")
    st.write(f"**Test Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**ROC AUC:** {roc_auc:.4f}")

    st.write("### Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual Negative", "Actual Positive"],
                          columns=["Predicted Negative", "Predicted Positive"]))

    # ====== 9. Training curve ======
    st.subheader("📈 Training and Validation Accuracy")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # ====== 10. Save results ======
    csv_path = "breast_cancer_model_comparison_results.csv"
    results_dict = {
        "model": "DeepLearning_Keras",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    if os.path.exists(csv_path):
        df_results = pd.read_csv(csv_path)
        df_results = pd.concat([df_results, pd.DataFrame([results_dict])], ignore_index=True)
    else:
        df_results = pd.DataFrame([results_dict])

    df_results.to_csv(csv_path, index=False)
    st.success(f"Results saved to {csv_path}")
