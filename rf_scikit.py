import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


"""
Random Forest pipeline for connectome analysis using scikit-learn
"""


def load_1D_as_matrix(file_path):
    return np.loadtxt(file_path, skiprows=1)


def compute_upper_triangle_features(matrix):
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    upper_idx = np.triu_indices_from(corr_matrix, k=1)
    return corr_matrix[upper_idx]


def generate_feature_names(n_rois):
    names = []
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            names.append(f"ROI{i}_ROI{j}")
    return names


def collect_files(asd_path, con_path):
    files = []

    for f in Path(asd_path).glob("*.1D"):
        files.append((f, 1))

    for f in Path(con_path).glob("*.1D"):
        files.append((f, 0))

    return files


def build_dataset(train_files,test_files):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    print("Processing Training files...")

    for file_path, label in train_files:
        matrix = load_1D_as_matrix(file_path)

        features = compute_upper_triangle_features(matrix)

        train_features.append(features)
        train_labels.append(label)

    X_train = np.array(train_features)
    y_train = np.array(train_labels)

    for file_path, label in test_files:
        matrix = load_1D_as_matrix(file_path)

        features = compute_upper_triangle_features(matrix)

        test_features.append(features)
        test_labels.append(label)

    X_test = np.array(test_features)
    y_test = np.array(test_labels)

    # Infer number of ROIs from feature count
    n_features = X_train.shape[1]
    # Solve n(n-1)/2 = n_features → approximate inverse
    n_rois = int((1 + np.sqrt(1 + 8 * n_features)) / 2)

    feature_names = generate_feature_names(n_rois)

    return X_train, y_train, X_test, y_test, feature_names


def train_random_forest(X_train, y_train, X_test, y_test, feature_names):

    # Model
    clf = RandomForestClassifier(
        n_estimators=1000,
        n_jobs=-1,
        random_state=42
    )

    print("Training Random Forest...")
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Feature importance
    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 20 Important Connections:")
    print(importance_df.head(20))

    # Save importance
    importance_df.to_csv("feature_importance.csv", index=False)

    return clf, importance_df


def main():
    train_subject_path = "C:/Users/chill/CS1980/CS1980-Brain-Graph-Analysis/Data/ML/Train/asd"
    train_con_path = "C:/Users/chill/CS1980/CS1980-Brain-Graph-Analysis/Data/ML/Train/control"
    test_subject_path = "C:/Users/chill/CS1980/CS1980-Brain-Graph-Analysis/Data/ML/Test/asd"
    test_con_path = "C:/Users/chill/CS1980/CS1980-Brain-Graph-Analysis/Data/ML/Test/control"

    train = collect_files(train_subject_path,train_con_path)
    test = collect_files(test_subject_path,test_con_path)

    if not train:
        print("No files found.")
        return
    if not test:
        print("No files found.")
        return

    X_train, y_train, X_test, y_test, feature_names = build_dataset(train,test)

    model, importance_df = train_random_forest(X_train, y_train, X_test, y_test, feature_names)

    print("\nSaved feature importances to feature_importance.csv")


if __name__ == "__main__":
    main()