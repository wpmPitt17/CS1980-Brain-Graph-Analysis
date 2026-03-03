import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report

# Path directory for all test and output files
test_path = './test_files'
asd_path = 'asd/rois_cc200'
control_path = 'control/rois_cc200'

out_path = './output_correlation'
out_control = 'output_control'
out_asd = 'output_asd'

rows = []
labels = []

# DX_GROUP 1 = Autism 2 = Control

if not os.path.exists(out_path):
    os.makedirs(out_path, exist_ok=True)

if not os.path.exists(os.path.join(out_path, out_asd)):
    os.makedirs(os.path.join(out_path, out_asd), exist_ok=True)


if not os.path.exists(os.path.join(out_path, out_control)):
    os.makedirs(os.path.join(out_path, out_control), exist_ok=True)

# For DX_GROUP 1
for filename in os.listdir(os.path.join(test_path, asd_path)):
    file_path = os.path.join(test_path, asd_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep='\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        rows.append(upper_triangle)
        labels.append(1)
        new_corr.to_csv(os.path.join(out_path, out_asd, filename))

# For DX_GROUP 2
for filename in os.listdir(os.path.join(test_path, control_path)):
    file_path = os.path.join(test_path, control_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep='\s+') 
        new_corr = df.corr(method='spearman')
        # Flatten correlation to 1d using upper triangle of matrix
        upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
        rows.append(upper_triangle)
        labels.append(2)
        new_corr.to_csv(os.path.join(out_path, out_control, filename))

X = np.array(rows) 
y = np.array(labels)

# Define Features of the Data and do train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y, shuffle=True)

X_test = np.nan_to_num(X_test)
X_train = np.nan_to_num(X_train)


# Initialize model and fit to the training data
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

report = classification_report(y_test, y_pred)
print(report)

confusion = confusion_matrix(y_test, y_pred, )
print(confusion_matrix)

# plt.figure(figsize=(10, 8))

# sns.heatmap(data=new_corr, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title(f"Spearman Correlation Heatmap for Last")

# plt.show()