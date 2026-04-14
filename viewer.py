import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, homogeneity_score, completeness_score, v_measure_score, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# Path directory for all test and output files
train_path = './data/ML/Train'
test_path = './data/ML/Test'
subject_path = 'asd'
control_path = 'control'

out_path = './output_correlation'
out_control = 'output_control'
out_asd = 'output_asd'

train_rows = []
train_labels = []
test_rows = []
test_labels = []

# DX_GROUP 0 = Autism, 1 = Control

if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)

if not os.path.exists(os.path.join(out_path, out_asd)):
    os.makedirs(os.path.join(out_path, out_asd), exist_ok=True)

if not os.path.exists(os.path.join(out_path, out_control)):
    os.makedirs(os.path.join(out_path, out_control), exist_ok=True)

# For DX_GROUP 1
for filename in os.listdir(os.path.join(train_path, subject_path)):
    file_path = os.path.join(train_path, subject_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep=r'\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        train_rows.append(upper_triangle)
        train_labels.append(0)
        new_corr.to_csv(os.path.join(out_path, out_asd, filename))

# For DX_GROUP 2
for filename in os.listdir(os.path.join(train_path, control_path)):
    file_path = os.path.join(train_path, control_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep=r'\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        train_rows.append(upper_triangle)
        train_labels.append(1)
        new_corr.to_csv(os.path.join(out_path, out_control, filename))

# For Test_GROUP 1
for filename in os.listdir(os.path.join(test_path, subject_path)):
    file_path = os.path.join(test_path, subject_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep=r'\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        test_rows.append(upper_triangle)
        test_labels.append(0)
        new_corr.to_csv(os.path.join(out_path, out_asd, filename))

# For Test_GROUP 2
for filename in os.listdir(os.path.join(test_path, control_path)):
    file_path = os.path.join(test_path, control_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep=r'\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        test_rows.append(upper_triangle)
        test_labels.append(1)
        new_corr.to_csv(os.path.join(out_path, out_control, filename))

X_train = np.array(train_rows) 
y_train = np.array(train_labels)

X_test = np.array(test_rows) 
y_test = np.array(test_labels)

X_test = np.nan_to_num(X_test)
X_train = np.nan_to_num(X_train)

# Scaler for standardizing features
scaler = StandardScaler()
pca = PCA(n_components=20)

# Reduce dimensionality of data for more efficient processing and reduce noise
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Initialize model and fit to the training data
# Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

model_scaled = LogisticRegression(max_iter=10000)
model_scaled.fit(X_train_scaled, y_train)

model_pca = LogisticRegression(max_iter=10000)
model_pca.fit(X_train_pca, y_train)

y_pred = model.predict(X_test)
y_pred_scaled = model_scaled.predict(X_test_scaled)
y_pred_pca = model_pca.predict(X_test_pca)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Logistic Regression Accuracy Scaled: {accuracy_score(y_test, y_pred_scaled):.4f}")
print(f"Logistic Regression Accuracy PCA: {accuracy_score(y_test, y_pred_pca):.4f}")

print(f"Logistic Regression Report: {classification_report(y_test, y_pred, target_names=['asd', 'Control'])}")
print(f"Logistic Regression Report Scaled: {classification_report(y_test, y_pred_scaled, target_names=['asd', 'Control'])}")
print(f"Logistic Regression Report PCA: {classification_report(y_test, y_pred_pca, target_names=['asd', 'Control'])}")

print(f"Logistic Regression Confusion: {confusion_matrix(y_test, y_pred)}")
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=["ASD", "Control"])
disp.plot(cmap=plt.cm.Blues)
plt.show()

print(f"Logistic Regression Confusion Scaled: {confusion_matrix(y_test, y_pred_scaled)}")
print(f"Logistic Regression Confusion PCA: {confusion_matrix(y_test, y_pred_pca)}")

# K means
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
cluster_labels = kmeans.fit_predict(X_train_pca)

cluster_labels_named = np.array(["ASD" if c == 0 else "Control" for c in cluster_labels])

print(kmeans)
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=cluster_labels_named, hue_order=["ASD", "Control"])
plt.title("KMeans (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# KMeans metrics
kmeans_mapped = cluster_labels
kmeans_accuracy = accuracy_score(y_train, kmeans_mapped)
print(f"KMeans Accuracy: {kmeans_accuracy:.4f}")
print(f"Homogeneity: {homogeneity_score(y_train, cluster_labels):.4f}")
print(f"Completeness: {completeness_score(y_train, cluster_labels):.4f}")
print(f"V-measure: {v_measure_score(y_train, cluster_labels):.4f}")
print(classification_report(y_train, kmeans_mapped, target_names=['ASD', 'Control']))

# KNN (try numerous k)
for k in range(3, 15, 2):
    neighbor = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    neighbor.fit(X_train_scaled, y_train)
    # print(neighbor.predict_proba(X_test))
    print(f"KNN Score Scaled: {neighbor.score(X_test_scaled, y_test)}")
for k in range(3, 15, 2):
    neighbor_pca = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    neighbor_pca.fit(X_train_pca, y_train)
    print(f"KNN Score PCA: {neighbor_pca.score(X_test_pca, y_test)}")