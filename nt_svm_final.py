import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline, defaultdict
from sklearn.utils import resample                      # downsample the dataset
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split    # split data into training and test sets
from sklearn.model_selection import GridSearchCV        # this will do cross validation
from sklearn.preprocessing import scale                 # scale and center the data
from sklearn.svm import SVC                             # this will make a support vector machine for classification
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, classification_report, f1_score, homogeneity_score, completeness_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score, roc_curve, v_measure_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, VarianceThreshold
from nilearn import datasets
from nilearn.image import get_data, index_img, load_img
from nilearn.plotting import find_probabilistic_atlas_cut_coords
from nilearn.image import coord_transform


# Path directory for all test and output files
train_path = './data/ML/Train'
test_path = './data/ML/Test'
eval_path = './data/ML/Eval'
subject_path = 'asd'
control_path = 'control'

out_path = './output_correlation/pearson'
out_control = 'output_control'
out_asd = 'output_asd'

def main():
    train_rows = []
    train_labels = []

    test_rows = []    
    test_labels = []

    eval_rows = []
    eval_labels = []

        
    check_dirs(out_path, out_asd, out_control, train_path, test_path, eval_path, subject_path, control_path)

    pearson_correlation(train_rows, train_labels, test_rows, test_labels, eval_rows, eval_labels)

    X_train, X_test, y_train, y_test, X_eval, y_eval = split(train_rows, test_rows, eval_rows, train_labels, test_labels, eval_labels)

    ensemble = svm(X_train, X_test, y_train, y_test, X_eval, y_eval)

    atlas_path = "cc200_roi_atlas.nii.gz"   # <-- your file
    atlas_img = load_img(atlas_path)
    atlas_data = get_data(atlas_img)
    roi_ids = np.unique(atlas_data)
    roi_ids = roi_ids[roi_ids > 0]   # remove background (0)
    roi_coords = []

    for roi_id in roi_ids:
        coords = np.column_stack(np.where(atlas_data == roi_id))
        
        # convert voxel indices → MNI space
        mni_coords = coord_transform(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            atlas_img.affine
        )
        
        # take mean (centroid)
        centroid = np.mean(np.array(mni_coords), axis=1)
        
        roi_coords.append(tuple(centroid))
    results = extract_key_features(ensemble, roi_coords)
    print(results)

    

def check_dirs(out_path, out_asd, out_control, train_path, test_path, eval_path, subject_path, control_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(os.path.join(out_path, out_asd)):
        os.makedirs(os.path.join(out_path, out_asd), exist_ok=True)

    if not os.path.exists(os.path.join(out_path, out_control)):
        os.makedirs(os.path.join(out_path, out_control), exist_ok=True)

    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)
    
    if not os.path.exists(os.path.join(train_path, subject_path)):
        os.makedirs(os.path.join(train_path, subject_path), exist_ok=True)
    
    if not os.path.exists(os.path.join(train_path, control_path)):
        os.makedirs(os.path.join(train_path, control_path), exist_ok=True)

    if not os.path.exists(test_path):
        os.makedirs(test_path, exist_ok=True)

    if not os.path.exists(os.path.join(test_path, subject_path)):
        os.makedirs(os.path.join(test_path, subject_path), exist_ok=True)
    
    if not os.path.exists(os.path.join(test_path, control_path)):
        os.makedirs(os.path.join(test_path, control_path), exist_ok=True)

    if not os.path.exists(eval_path):
        os.makedirs(eval_path, exist_ok=True)

    if not os.path.exists(os.path.join(eval_path, subject_path)):
        os.makedirs(os.path.join(eval_path, subject_path), exist_ok=True)
    
    if not os.path.exists(os.path.join(eval_path, control_path)):
        os.makedirs(os.path.join(eval_path, control_path), exist_ok=True)


def pearson_correlation(train_rows, train_labels, test_rows, test_labels, eval_rows, eval_labels):
    # DX_GROUP 0 = Autism, 1 = Control
    # For DX_GROUP 1
    for filename in os.listdir(os.path.join(train_path, subject_path)):
        file_path = os.path.join(train_path, subject_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            train_rows.append(upper_triangle)
            train_labels.append(0)
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

    # For DX_GROUP 2
    for filename in os.listdir(os.path.join(train_path, control_path)):
        file_path = os.path.join(train_path, control_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            train_rows.append(upper_triangle)
            train_labels.append(1)
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

    # For Test_GROUP 1
    for filename in os.listdir(os.path.join(test_path, subject_path)):
        file_path = os.path.join(test_path, subject_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            test_rows.append(upper_triangle)
            test_labels.append(0)
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

    # For Test_GROUP 2
    for filename in os.listdir(os.path.join(test_path, control_path)):
        file_path = os.path.join(test_path, control_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            test_rows.append(upper_triangle)
            test_labels.append(1)
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

        # For Eval_GROUP 1
    for filename in os.listdir(os.path.join(eval_path, subject_path)):
        file_path = os.path.join(eval_path, subject_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            eval_rows.append(upper_triangle)
            eval_labels.append(0)
            # new_corr.to_csv(os.path.join(out_path, out_asd, filename))

    # For Eval_GROUP 2
    for filename in os.listdir(os.path.join(eval_path, control_path)):
        file_path = os.path.join(eval_path, control_path, filename)

        with open(file_path, 'r') as f:
            df = pd.read_csv(f, sep=r'\s+') 
            new_corr = df.corr(method='pearson')
            # Flatten correlation to 1d using upper triangle of matrix
            upper_triangle = new_corr.values[np.triu_indices(200, k=1)]
            upper_triangle = np.nan_to_num(upper_triangle)
            upper_triangle = np.clip(upper_triangle, -0.999999, 0.999999)
            upper_triangle = np.arctanh(upper_triangle)
            eval_rows.append(upper_triangle)
            eval_labels.append(1)
            # new_corr.to_csv(os.path.join(out_path, out_control, filename))

def split(train_rows, test_rows, eval_rows, train_labels, test_labels, eval_labels):
    X_train = np.array(train_rows) 
    y_train = np.array(train_labels)

    X_test = np.array(test_rows) 
    y_test = np.array(test_labels)

    X_eval = np.array(eval_rows)
    y_eval = np.array(eval_labels)

    return X_train, X_test, y_train, y_test, X_eval, y_eval

def svm(X_train, X_test, y_train, y_test, X_eval, y_eval):

    # Parameter tuning grid for SVM using PCA for dimensionality reduction
    #   tests different numbers of PCA components (both fixed and variance-based) and both linear and RBF kernels with various C and gamma values
    # pipelinePCA = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('pca', PCA()),
    #     ('svm', SVC(class_weight='balanced'))
    # ])

    # param_gridPCA = [
    #     # Linear SVM
    #     {
    #         'pca__n_components': [50, 100, 150, 200],
    #         'pca__n_components': [0.90, 0.95, 0.99],
    #         'svm__kernel': ['linear'],
    #         'svm__C': [0.01, 0.1, 1, 10, 50]
    #     },
    #     # RBF SVM
    #     {
    #         'pca__n_components': [50, 100, 150],
    #         'pca__n_components': [0.90, 0.95, 0.99],
    #         'svm__kernel': ['rbf'],
    #         'svm__C': [0.1, 1, 10, 50],
    #         'svm__gamma': ['scale', 0.01, 0.001, 0.0001]
    #     }

    # pipeline_rfe = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('rfe', RFE(
    #         estimator=RidgeClassifier(),
    #         step=1000
    #     )),
    #     ('svm', SVC(kernel='linear'))
    # ])

    # param_grid_rfe = {
    #     'rfe__n_features_to_select': [200, 500, 1000],
    #     'svm__C': [0.1, 1, 10]
    # }
    #     #Linear SVM
    #     {
    #         'rfe__n_features_to_select': [200, 500, 1000],
    #         'svm__kernel': ['linear'],
    #         'svm__C': [0.01, 0.1, 1, 10, 50]
    #     },
    #     # RBF SVM
    #     {
    #         'rfe__n_features_to_select': [200, 500, 1000],
    #         'svm__kernel': ['rbf'],
    #         'svm__C': [0.1, 1, 10, 50],
    #         'svm__gamma': ['scale', 0.01, 0.001, 0.0001]
    #     }

    # grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)

    # grid.fit(X_train, y_train)

    # y_pred = grid.predict(X_test)


    # Pipelines with optimized parameters hardcoded based on previous grid search results to save time

    pipeline_pca = Pipeline([
        ('var', VarianceThreshold(1e-5)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('svm', SVC(kernel='linear', C=0.01, class_weight='balanced',  probability=True))
    ])

    pipeline_rfe = Pipeline([
        ('scaler', StandardScaler()),
        ('rfe', RFE(RidgeClassifier(), n_features_to_select=500, step=2000)),
        ('svm', SVC(kernel='linear', C=1, class_weight='balanced', probability=True))
    ])

    ensemble = VotingClassifier([
            ('pca', pipeline_pca),
            ('rfe', pipeline_rfe)
        ], 
        voting='soft',
        weights=[2, 1]   # PCA counts twice as much. Doing this because rfe produced a high test accuracy (>0.72) but a much lower eval accuracy (<0.62) 
    )

    ensemble.fit(X_train, y_train)

    print("Ensemble model metrics:")
    y_test_pred = ensemble.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))

    y_eval_pred = ensemble.predict(X_eval)
    print("Eval accuracy:", accuracy_score(y_eval, y_eval_pred))
    print("Precision: ", precision_score(y_test, y_test_pred))
    print("Recall score: ", recall_score(y_test, y_test_pred))
    print('F1 SCORE: ', f1_score(y_test, y_test_pred))

    # Confusion matrix for test set
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_test_pred), display_labels=["ASD", "Control"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.show()

    test_results_prob = ensemble.predict_proba(X_test)

    # ROC curve and AUC for test set
    # test_fpr, test_tpr, te_thresholds = roc_curve(y_test, test_results_prob[:,1],pos_label=1)
    # plt.grid()
    # plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
    # plt.plot([0,1],[0,1],'g--')
    # plt.legend()
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Test AUC(ROC curve)")
    # plt.grid(color='black', linestyle='-', linewidth=0.5)
    # plt.show()

    return ensemble


def extract_key_features(ensemble, roi_coords, top_k_edges=50, top_k_rois=20):
    # ---------------------------
    # Build ROI pairs
    # ---------------------------
    roi_pairs = [(i, j) for i in range(200) for j in range(i+1, 200)]

    # ===========================
    # RFE FEATURES
    # ===========================
    rfe_model = ensemble.named_estimators_['rfe']
    rfe = rfe_model.named_steps['rfe']
    selected_mask = rfe.support_

    selected_features = [roi_pairs[i] for i in range(len(roi_pairs)) if selected_mask[i]]

    svm_rfe = rfe_model.named_steps['svm']
    weights = svm_rfe.coef_.flatten()

    # Rank only selected features
    rfe_scores = np.abs(weights)
    top_rfe_idx = np.argsort(rfe_scores)[-top_k_edges:]
    top_rfe_features = [selected_features[i] for i in top_rfe_idx]

    # ===========================
    # PCA FEATURES
    # ===========================
    pca_model = ensemble.named_estimators_['pca']
    pca = pca_model.named_steps['pca']

    importance = np.sum(np.abs(pca.components_), axis=0)

    top_pca_idx = np.argsort(importance)[-top_k_edges:]
    top_pca_features = [roi_pairs[i] for i in top_pca_idx]

    # ===========================
    # COMBINE FEATURES
    # ===========================
    combined_score = importance.copy()

    # boost RFE-selected features
    for i, selected in enumerate(selected_mask):
        if selected:
            combined_score[i] *= 2

    top_combined_idx = np.argsort(combined_score)[-top_k_edges:]
    top_combined_features = [roi_pairs[i] for i in top_combined_idx]

    # ===========================
    # ROI IMPORTANCE
    # ===========================
    roi_importance = defaultdict(float)

    for idx, (i, j) in enumerate(roi_pairs):
        weight = combined_score[idx]
        roi_importance[i] += weight
        roi_importance[j] += weight
    
    n_rois = 200
    edges_per_roi = n_rois - 1

    for i in roi_importance:
        roi_importance[i] /= edges_per_roi

    sorted_rois = sorted(roi_importance.items(), key=lambda x: x[1], reverse=True)
    top_rois = sorted_rois[:top_k_rois]

    # ===========================
    # ATLAS MAPPING
    # ===========================

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    labels = atlas.labels
    atlas_img = load_img(atlas.maps)
    atlas_data = get_data(atlas_img)
    def get_region_label(x, y, z):
        coords = np.round(
            coord_transform(x, y, z, np.linalg.inv(atlas_img.affine))
        ).astype(int)

        try:
            label_idx = atlas_data[coords[0], coords[1], coords[2]]
            return labels[label_idx]
        except:
            return "Unknown"

    roi_labels = {}
    for i, (x, y, z) in enumerate(roi_coords):
        roi_labels[i] = get_region_label(x, y, z)

    top_regions_named = [(roi_labels[i], score) for i, score in top_rois]

    # ===========================
    # OUTPUT
    # ===========================
    print("\n=== TOP RFE FEATURES ===")
    print(top_rfe_features)

    print("\n=== TOP PCA FEATURES ===")
    print(top_pca_features)

    print("\n=== TOP COMBINED FEATURES ===")
    print(top_combined_features)

    print("\n=== TOP BRAIN REGIONS ===")
    for region, score in top_regions_named:
        print(f"{region}: {score:.2f}")

    


    yeo = datasets.fetch_atlas_yeo_2011()
    maps_img = load_img(yeo['maps'])
    yeo_img = index_img(maps_img, 0)
    yeo_data = get_data(yeo_img)

    network_names = {
        0: "Background",
        1: "Visual",
        2: "Somatomotor",
        3: "Dorsal Attention",
        4: "Salience (Ventral Attention)",
        5: "Limbic",
        6: "Frontoparietal",
        7: "Default Mode"
    }

    def get_network_label(x, y, z):
        coords = np.round(
            coord_transform(x, y, z, np.linalg.inv(yeo_img.affine))
        ).astype(int)

        try:
            label = yeo_data[coords[0], coords[1], coords[2]]
            return network_names.get(label, "Unknown")
        except:
            return "Unknown"
        
    roi_to_network = {}
    for i, (x, y, z) in enumerate(roi_coords):
        roi_to_network[i] = get_network_label(x, y, z)

    network_importance = defaultdict(float)

    for roi, score in roi_importance.items():
        net = roi_to_network[roi]
        network_importance[net] += score

    total = sum(network_importance.values())

    for net in network_importance:
        network_importance[net] /= total

    sorted_networks = sorted(network_importance.items(), key=lambda x: x[1], reverse=True)

    print("\n=== NETWORK IMPORTANCE ===")
    for net, score in sorted_networks:
        print(f"{net}: {score:.3f}")
    

    return {
            "rfe_features": top_rfe_features,
            "pca_features": top_pca_features,
            "combined_features": top_combined_features,
            "top_regions": top_regions_named
        }

if __name__ == "__main__":
    main()
