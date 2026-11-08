"""Streamlit app: evaluate the saved phishing classifiers on an uploaded test CSV (no retraining)."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics import METRIC_NAMES, POSITIVE_LABEL, compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_FILE = ROOT / "test_data.csv"
TARGET = "status"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MODEL_INFO = {
    "Logistic Regression": "Linear model on scaled features. Fast, simple baseline.",
    "Decision Tree": "Rule-based splits on URL and page features (max depth 10).",
    "KNN": "Labels a page by its 7 closest neighbours in the training data.",
    "Naive Bayes": "Gaussian Naive Bayes. Assumes features are independent.",
    "Support Vector Machine": "RBF-kernel SVM on scaled features.",
}

st.set_page_config(page_title="Phishing Website Detector", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data(show_spinner=False)
def evaluate(name, data):
    """Score one saved model on the uploaded data. Returns (y_true, y_pred, scores)."""
    model = load_model(name)
    X = data[list(model.feature_names_in_)]
    y_true = data[TARGET].to_numpy()
    y_pred = model.predict(X)
    phishing_col = list(model.classes_).index(POSITIVE_LABEL)
    proba = model.predict_proba(X)[:, phishing_col]
    return y_true, y_pred, compute_metrics(y_true, y_pred, proba)


def validate(data):
    """Return an error message if the CSV cannot be scored, otherwise None."""
    model = load_model("Logistic Regression")
    features = list(model.feature_names_in_)
    missing = sorted(set(features + [TARGET]) - set(data.columns))
    if missing:
        return f"Missing required columns: {', '.join(missing)}"
    if data.empty:
        return "The file has no rows."
    if data[features + [TARGET]].isna().any().any():
        return "The file has empty cells. Fill or remove them and upload again."
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(data[c])]
    if bad:
        return f"These feature columns must be numeric: {', '.join(bad)}"
    unknown = sorted(set(data[TARGET].astype(str)) - set(model.classes_))
    if unknown:
        return f"Unknown values in '{TARGET}': {', '.join(unknown)}. Expected: {', '.join(model.classes_)}."
    return None


def confusion_figure(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=colour, fontsize=13)
    fig.tight_layout()
    return fig


def comparison_figure(table, metric, selected):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    values = table[metric].fillna(0)
    colours = ["#1f6feb" if name == selected else "#9ec1f7" for name in table.index]
    ax.barh(table.index, values, color=colours)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    for y_pos, value in enumerate(values):
        ax.text(min(value + 0.01, 0.9), y_pos, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def sidebar():
    st.sidebar.header("1. Test data")
    upload = st.sidebar.file_uploader("Upload test CSV", type="csv")
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=upload is None)
    if SAMPLE_FILE.exists():
        st.sidebar.download_button("Download sample test CSV", SAMPLE_FILE.read_bytes(), "test_data.csv", "text/csv")

    st.sidebar.header("2. Model")
    choice = st.sidebar.selectbox("Select classifier", list(MODEL_FILES))
    st.sidebar.caption(MODEL_INFO[choice])

    if upload is not None:
        try:
            return pd.read_csv(upload), upload.name, choice
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            st.sidebar.error("That file could not be read as a CSV.")
            return None, None, choice
    if use_sample and SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE), SAMPLE_FILE.name, choice
    return None, None, choice


def main():
    st.title("Phishing Website Detector")
    st.write(
        "Pre-trained classifiers that label a web page as **phishing** or **legitimate** from "
        "87 URL, content and domain features. Upload a test CSV and pick a model: the saved "
        "model is only evaluated, never retrained."
    )

    data, source, choice = sidebar()
    if data is None:
        st.info("Upload a test CSV in the sidebar, or tick the bundled test file.")
        return
    error = validate(data)
    if error:
        st.error(error)
        return

    st.caption(
        f"Source: {source} | rows: {len(data):,} | columns: {data.shape[1]} | "
        "The 'url' column (if present) is shown for reference only and is not used by the models."
    )

    table = pd.DataFrame({name: evaluate(name, data)[2] for name in MODEL_FILES}).T[METRIC_NAMES]
    table.index.name = "ML Model Name"
    y_true, y_pred, scores = evaluate(choice, data)

    tab_eval, tab_compare, tab_data = st.tabs(["Evaluation", "Model comparison", "Data preview"])

    with tab_eval:
        st.subheader(f"Evaluation metrics: {choice}")
        for col, metric in zip(st.columns(len(METRIC_NAMES)), METRIC_NAMES):
            value = scores[metric]
            col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            labels = sorted(load_model(choice).classes_)
            st.pyplot(confusion_figure(y_true, y_pred, labels))
        with right:
            st.subheader("Classification report")
            report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            report.pop("accuracy", None)
            frame = pd.DataFrame(report).T
            frame["support"] = frame["support"].astype(int)
            st.dataframe(frame.round(4))
            st.download_button(
                "Download predictions (CSV)",
                data.assign(Predicted=y_pred).to_csv(index=False),
                f"predictions_{choice.lower().replace(' ', '_')}.csv",
                "text/csv",
            )

    with tab_compare:
        st.subheader("All models on the same test file")
        metric = st.selectbox("Metric to plot", METRIC_NAMES)
        left, right = st.columns([1, 1.2])
        with left:
            st.pyplot(comparison_figure(table, metric, choice))
        with right:
            st.dataframe(table.style.format("{:.4f}", na_rep="n/a"))
            st.caption(f"Precision, Recall and F1 treat '{POSITIVE_LABEL}' as the positive class.")

    with tab_data:
        st.subheader("Class balance")
        st.bar_chart(data[TARGET].value_counts())
        st.subheader("First 100 rows")
        st.dataframe(data.head(100))


main()
 