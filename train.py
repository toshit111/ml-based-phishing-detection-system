"""Train the five classifiers on the phishing dataset, save them and write test_data.csv."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics import POSITIVE_LABEL, compute_metrics

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "dataset_phishing.csv"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
TARGET = "status"
ID_COLUMN = "url"  # kept in the CSVs for readability, never used as a model feature
SEED = 42
TEST_FRACTION = 0.20


def build_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED)
        ),
    }


def model_file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = pd.read_csv(RAW_FILE)
    # Duplicates are judged on the numeric features + target only (url is ignored).
    df = df.drop_duplicates(subset=[c for c in df.columns if c != ID_COLUMN]).reset_index(drop=True)
    X_all, y = df.drop(columns=TARGET), df[TARGET]
    print(f"rows={len(df)}  features={X_all.shape[1] - 1}  classes={sorted(y.unique())}")

    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED
    )
    print(f"train={len(X_train_all)}  test={len(X_test_all)}")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    # test_data.csv keeps the url column (upload-friendly); models never see it.
    pd.concat([X_test_all, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)

    X_train = X_train_all.drop(columns=ID_COLUMN)
    X_test = X_test_all.drop(columns=ID_COLUMN)

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / model_file_name(name))
        phishing_col = list(model.classes_).index(POSITIVE_LABEL)
        scores = compute_metrics(
            y_test, model.predict(X_test), model.predict_proba(X_test)[:, phishing_col]
        )
        rows.append({"ML Model Name": name, **scores})
        print(f"trained {name:24s} accuracy={scores['Accuracy']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
 