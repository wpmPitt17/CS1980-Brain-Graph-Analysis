import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report, homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import silhouette_samples, silhouette_score

# Path directory for all test and output files
train_path = './data/ML/Train'
test_path = './data/ML/Test'
subject_path = 'asd'
control_path = 'control'

out_path = './output_correlation'
out_control = 'output_control'
out_asd = 'output_asd'

def main():
    train_rows = []
    train_labels = []
    test_rows = []
    test_labels = []
        
    check_dirs(out_path, out_asd, out_control)

    convert_to_correlation(train_rows, train_labels, test_rows, test_labels)

    X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca, X_test_pca = split(train_rows, train_labels, test_rows, test_labels)

    logistic_regression(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca, X_test_pca)
    kmean(X_train_pca, y_train)
    knn(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca,X_test_pca)
    kfoldLR()
    finetuneLR()


def check_dirs(out_path, out_asd, out_control):
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(os.path.join(out_path, out_asd)):
        os.makedirs(os.path.join(out_path, out_asd), exist_ok=True)

    if not os.path.exists(os.path.join(out_path, out_control)):
        os.makedirs(os.path.join(out_path, out_control), exist_ok=True)

def convert_to_correlation(train_rows, train_labels, test_rows, test_labels):
    # DX_GROUP 0 = Autism, 1 = Control
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
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

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
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

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
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

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
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

def split(train_rows, train_labels, test_rows, test_labels):
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

    return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca,X_test_pca

def logistic_regression(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca, X_test_pca):
    print('===========================\n Logistic Regression Metrics\n===========================')
    
    # Initialize model and fit to the training data
    # Logistic Regression
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    model_scaled = LogisticRegression(max_iter=2000)
    model_scaled.fit(X_train_scaled, y_train)

    model_pca = LogisticRegression(max_iter=2000)
    model_pca.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test)
    y_pred_scaled = model_scaled.predict(X_test_scaled)
    y_pred_pca = model_pca.predict(X_test_pca)

    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Logistic Regression Accuracy Scaled: {accuracy_score(y_test, y_pred_scaled):.4f}")
    print(f"Logistic Regression Accuracy PCA: {accuracy_score(y_test, y_pred_pca):.4f}")

    print(f"Logistic Regression Report: {classification_report(y_test, y_pred, target_names=["ASD", "Control"])}")
    print(f"Logistic Regression Report Scaled: {classification_report(y_test, y_pred_scaled, target_names=["ASD", "Control"])}")
    print(f"Logistic Regression Report PCA: {classification_report(y_test, y_pred_pca, target_names=["ASD", "Control"])}")

    print(f"Logistic Regression Confusion: {confusion_matrix(y_test, y_pred)}")
    print(f"Logistic Regression Confusion Scaled: {confusion_matrix(y_test, y_pred_scaled)}")
    print(f"Logistic Regression Confusion PCA: {confusion_matrix(y_test, y_pred_pca)}")

def kmean(X_train_pca, y_train):
    print('===========================\n K-Means Metrics\n===========================')
    cluster_range = [2, 3, 4, 5, 6, 7]

    # Get silhouette score for different number of clusters
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_train_pca)
        
        silhouetter_avg = silhouette_score(X_train_pca, cluster_labels)
        print(f"For {n_clusters} clusters, silhouette avg score is: {silhouetter_avg}")

    # K means for n=2 (most accurate value for k)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_train_pca)

    cluster_labels_named = np.array(["ASD" if c == 0 else "Control" for c in cluster_labels])

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
    print(f"KMeans Report:\n{classification_report(y_train, kmeans_mapped, target_names=["ASD", "Control"])}")

def knn(X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, X_train_pca,X_test_pca):
    print('===========================\n KNN Metrics\n===========================')

    # KNN (try numerous k)
    for k in range(3, 9, 2):
        neighbor = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
        neighbor.fit(X_train, y_train)
        y_pred_knn = neighbor.predict(X_test)
        print(f"KNN (k={k}) Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
        print(f"KNN (k={k}) Report:\n{classification_report(y_test, y_pred_knn, target_names=['ASD', 'Control'])}")
    for k in range(3, 9, 2):
        neighbor = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
        neighbor.fit(X_train_scaled, y_train)
        y_pred_knn_scaled = neighbor.predict(X_test_scaled)
        print(f"KNN Scaled (k={k}) Accuracy: {accuracy_score(y_test, y_pred_knn_scaled):.4f}")
        print(f"KNN Scaled (k={k}) Report:\n{classification_report(y_test, y_pred_knn_scaled, target_names=['ASD', 'Control'])}")
        # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred_knn_scaled), display_labels=["ASD", "Control"])
        # disp.plot(cmap=plt.cm.Blues)
        # plt.show()
    for k in range(3, 9, 2):
        neighbor_pca = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
        neighbor_pca.fit(X_train_pca, y_train)
        y_pred_knn_pca = neighbor_pca.predict(X_test_pca)
        print(f"KNN PCA (k={k}) Accuracy: {accuracy_score(y_test, y_pred_knn_pca):.4f}")
        print(f"KNN PCA (k={k}) Report:\n{classification_report(y_test, y_pred_knn_pca, target_names=['ASD', 'Control'])}")


def kfoldLR():
    print('===========================\nK-Fold Logistic Regression Metrics\n===========================')
    all_data = collect_all_subjects() 
    
    dataX = np.array(all_data[0])
    dataY = np.array(all_data[1])

    dataX = np.nan_to_num(dataX)

    mean_accuracy_norm = 0
    mean_accuracy_scaled = 0
    mean_accuracy_rfe = 0

    k = 10
    kfold = KFold(n_splits=k, shuffle=True, random_state=1)
    for train, test in kfold.split(dataX):
        # Logistic regression and average accuracy based on rfe
        lrmodel = LogisticRegression(max_iter=2000)
        rfe = RFE(estimator=lrmodel, n_features_to_select=900, step=400)
        rfe.fit(dataX[train], dataY[train])
        y_pred_rfe = rfe.predict(dataX[test])
        mean_accuracy_rfe += accuracy_score(dataY[test], y_pred_rfe)
        
        # Logistic regression and average accuracy
        lrmodel.fit(dataX[train], dataY[train])
        y_pred = lrmodel.predict(dataX[test])
        mean_accuracy_norm += accuracy_score(dataY[test], y_pred)

        # Logistic regression and average accuracy based on scaled values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(dataX[train])
        X_test_scaled = scaler.fit_transform(dataX[test])
        lrmodel.fit(X_train_scaled, dataY[train])
        y_pred_scaled = lrmodel.predict(X_test_scaled)
        mean_accuracy_scaled += accuracy_score(dataY[test], y_pred_scaled)

    print(f"{k}-Fold Cross Validation Logistic Regression\nMean Accuracy: {mean_accuracy_norm / k}") 
    print(f"{k}-Fold Cross Validation Scaled Logistic Regression\nMean Accuracy: {mean_accuracy_scaled / k}") 
    print(f"{k}-Fold Cross Validation Logistic Regression with RFE\nMean Accuracy: {mean_accuracy_rfe / k}") 

    print('===========================')

def finetuneLR():
    print('===========================\nParameter Fine-Tuning Logistic Regression Metrics\n===========================')
    all_data = collect_all_subjects()

    dataX = np.array(all_data[0])
    dataY = np.array(all_data[1])

    dataX = np.nan_to_num(dataX)

    mean_accuracy_norm = 0
    mean_accuracy_scaled = 0

    param_grid = {'C' : [0.001,0.01,0.1,1,10,100,1000]}

    clf = GridSearchCV(LogisticRegression(max_iter=2000), param_grid)
    clf.fit(dataX,dataY)

    # Logistic regression and average accuracy based on scaled values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(dataX)


    clf_scaled = GridSearchCV(LogisticRegression(max_iter=2000),param_grid)
    clf_scaled.fit(X_train_scaled,dataY)

    print("Logistic Regression with Grid Search Parameter Optimization")
    print(clf.best_estimator_)
    print(clf.best_params_)
    print(clf.best_score_)
    print("Logistic Regression with Grid Search Parameter Optimization and Scaled")
    print(clf_scaled.best_estimator_)
    print(clf_scaled.best_params_)
    print(clf_scaled.best_score_)
    print('===========================')

def collect_all_subjects():
    # all_data[0] stores correlations, all_data[1] stores labels
    all_data = ([], [])
    
    # Path directory for all test data
    all_data_path = './data/1D'
    subject_path = 'asd/rois_cc200'
    control_path = 'control/rois_cc200'

    # DX_GROUP 0 = Autism, 1 = Control
    # For DX_GROUP 1
    for filename in os.listdir(os.path.join(all_data_path, subject_path)):
        file_path = os.path.join(all_data_path, subject_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='spearman')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            all_data[0].append(upper_triangle)
            all_data[1].append(0)
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

    # For DX_GROUP 2
    for filename in os.listdir(os.path.join(all_data_path, control_path)):
        file_path = os.path.join(all_data_path, control_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='spearman')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            all_data[0].append(upper_triangle)
            all_data[1].append(1)
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

    return all_data

if __name__ == "__main__":
    main()