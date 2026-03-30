import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Path directory for all control, patient, and output files
data_path = './data/1D'
asd_path = 'asd/rois_cc200'
control_path = 'control/rois_cc200'

train_path = './data/ML/Train'
test_path = './data/ML/Test'
eval_path = './data/ML/Eval'

output_subject = 'asd'
output_control = 'control'

rows = []
labels = []

# DX_GROUP 0 = Autism, 1 = Control

if not os.path.exists(train_path):
    os.makedirs(train_path, exist_ok=True)

if not os.path.exists(os.path.join(train_path, output_subject)):
    os.makedirs(os.path.join(train_path, output_subject), exist_ok=True)


if not os.path.exists(os.path.join(train_path, output_control)):
    os.makedirs(os.path.join(train_path, output_control), exist_ok=True)

if not os.path.exists(test_path):
    os.makedirs(test_path, exist_ok=True)

if not os.path.exists(os.path.join(test_path, output_subject)):
    os.makedirs(os.path.join(test_path, output_subject), exist_ok=True)


if not os.path.exists(os.path.join(test_path, output_control)):
    os.makedirs(os.path.join(test_path, output_control), exist_ok=True)

if not os.path.exists(eval_path):
    os.makedirs(eval_path, exist_ok=True)

if not os.path.exists(os.path.join(eval_path, output_subject)):
    os.makedirs(os.path.join(eval_path, output_subject), exist_ok=True)


if not os.path.exists(os.path.join(eval_path, output_control)):
    os.makedirs(os.path.join(eval_path, output_control), exist_ok=True)

# For DX_GROUP 1
for filename in os.listdir(os.path.join(data_path, asd_path)):
    file_path = os.path.join(data_path, asd_path, filename)

    rows.append(file_path)
    labels.append(1)

# For DX_GROUP 2
for filename in os.listdir(os.path.join(data_path, control_path)):
    file_path = os.path.join(data_path, control_path, filename)

    rows.append(file_path)
    labels.append(0)

X = np.array(rows) 
y = np.array(labels)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y, shuffle=True)

for path, y in zip(X_train, y_train):
    if y == 1:
        try:
            shutil.copy(path, os.path.join(train_path,output_subject))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        try:
            shutil.copy(path, os.path.join(train_path,output_control))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp, shuffle=True)

for path, y in zip(X_test, y_test):
    if y == 1:
        try:
            shutil.copy(path, os.path.join(test_path,output_subject))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        try:
            shutil.copy(path, os.path.join(test_path,output_control))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")

for path, y in zip(X_val, y_val):
    if y == 1:
        try:
            shutil.copy(path, os.path.join(eval_path,output_subject))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        try:
            shutil.copy(path, os.path.join(eval_path,output_control))
        except shutil.SameFileError:
            print("Source and destination represent the same file.")
        except PermissionError:
            print("Permission denied.")
        except FileNotFoundError:
            print(f"Source file not found for file {path}")
        except Exception as e:
            print(f"An error occurred: {e}")