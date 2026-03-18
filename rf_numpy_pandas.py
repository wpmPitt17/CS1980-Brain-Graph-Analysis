import numpy as np
import pandas as pd
from pathlib import Path
import subprocess


"""
Optimized version:
- Uses NumPy for fast correlation computation
- Uses Pandas for writing structured data
- Much faster than manual Pearson loops
"""


def load_1D_as_matrix(file_path):
    """
    Load .1D file into a NumPy array (rows = timepoints, cols = ROIs)
    Skips header automatically.
    """
    return np.loadtxt(file_path, skiprows=1)


def compute_upper_triangle_features(matrix):
    """
    Compute correlation matrix and return upper triangle (excluding diagonal)
    """
    corr_matrix = np.corrcoef(matrix, rowvar=False)

    # Get upper triangle indices
    upper_idx = np.triu_indices_from(corr_matrix, k=1)

    return corr_matrix[upper_idx]


def generate_feature_names(n_cols):
    """
    Generate feature names like ROI0_ROI1, ROI0_ROI2, ...
    """
    names = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            names.append(f"ROI{i}_ROI{j}")
    return names


def collect_files(asd_path, con_path):
    """
    Collect files with labels
    ASD = 1, Control = 0
    """
    files = []

    for f in Path(asd_path).glob("*.1D"):
        files.append((f, 1))

    for f in Path(con_path).glob("*.1D"):
        files.append((f, 0))

    return files


def build_dataset(file_list):
    """
    Build full dataset as Pandas DataFrame
    """
    all_features = []
    labels = []

    print("Processing files...")

    for file_path, label in file_list:
        matrix = load_1D_as_matrix(file_path)

        features = compute_upper_triangle_features(matrix)

        all_features.append(features)
        labels.append(label)

    X = np.array(all_features)
    y = np.array(labels)

    # Generate column names
    feature_names = generate_feature_names(X.shape[1] + 1)

    df = pd.DataFrame(X, columns=feature_names)
    df["Diagnosis"] = y

    return df


def main():
    asd_path = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/1D/ASD/Outputs/cpac/filt_global/rois_ho"
    con_path = "/home/rhonda/Downloads/capstone/Brain-Analysis/data/1D/Control/Outputs/cpac/filt_global/rois_ho"

    output_file = "connectome_data.dat"

    files = collect_files(asd_path, con_path)

    if not files:
        print("No files found.")
        return

    df = build_dataset(files)

    # Save to file (same format Ranger expects)
    df.to_csv(output_file, index=False)

    print(f"Successfully created {output_file}")

    # Run Ranger
    ranger_cmd = [
        "/home/rhonda/Downloads/capstone/ranger/cpp_version/build/ranger",
        "--file", output_file,
        "--depvarname", "Diagnosis",
        "--treetype", "1",
        "--ntree", "1000",
        "--nthreads", "4",
        "--impmeasure", "1",
        "--verbose"
    ]

    print("Starting Ranger Random Forest...")
    result = subprocess.run(ranger_cmd)

    if result.returncode == 0:
        print("Ranger finished successfully. Check ranger_importance.out for results!")


if __name__ == "__main__":
    main()