#Imports
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
import geomstats.datasets.utils as data_utils 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns



def load_split(root_dir):
    data = []
    labels = []
    ids = []

    label_map = {
        "control": 0,
        "asd": 1
    }

    for label_name in ["control", "asd"]:
        folder = os.path.join(root_dir, label_name)

        for file in tqdm(os.listdir(folder), desc=f"Loading {label_name}"):
            if not file.endswith(".1D"):
                continue

            file_path = os.path.join(folder, file)

            df = pd.read_csv(file_path, sep=r'\s+', header=None, comment='#')

            corr = df.corr(method='spearman')

            data.append(corr.values)
            labels.append(label_map[label_name])
            ids.append(file)

    return np.array(data), np.array(labels), ids

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=200, out_channels=16)
        self.conv2 = GCNConv(in_channels=16, out_channels=8)
        self.lin = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x.view(-1)

def to_graph_list(X, y):
    data_list = []
    N = X.shape[1]

    for i in range(len(X)):
        matrix = X[i]
        threshold = 0.3
        mask = np.abs(matrix) > threshold

        edge_idx = np.array(mask.nonzero())
        
        if edge_idx.shape[1] == 0:
            continue  # skip bad samples

        edge_index = torch.tensor(edge_idx, dtype=torch.long).t()

        edge_attr = matrix[edge_index[0], edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        if edge_attr.numel() > 0:
            edge_attr = edge_attr / edge_attr.abs().max()

        x = torch.tensor(matrix, dtype=torch.float32)

        graph = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y[i].unsqueeze(0)
        )

        data_list.append(graph)

    return data_list

# ==== Main ====

def main():

    train_dir = input("Enter train data file path: ")
    test_dir = input("Enter test data file path: ")
    
    X_train, y_train, train_ids = load_split(train_dir) # data, labels, ids
    X_test, y_test, test_ids = load_split(test_dir)
    
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)


    # Cleaning data
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()


    train_data = to_graph_list(X_train, y_train)
    test_data = to_graph_list(X_test, y_test)

    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    
    def train():
        model.train()
        total_loss = 0
        for data in train_loader:
            if torch.isnan(data.x).any():
                continue
            if torch.isnan(data.edge_attr).any():
                continue
            optimizer.zero_grad()
            out = model(data)
            if torch.isnan(out).any():
                continue
            loss = criterion(out, data.y)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader)
    
    losses = []
    for epoch in range(100):
        loss = train()
        losses.append(loss)
        if epoch%10 == 0:
            print(f'Epoch {epoch}, Loss {loss:.4f}')
    epochs = list(range(1, len(losses)+1))
    
    plt.figure(figsize=(6,3))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    model.eval()
    actual_labels = []
    predicted_labels= []
    
    correct = 0
    
    for data in test_loader:
        out = model(data)
        pred = torch.sigmoid(out)>0.5
        correct += (pred == data.y).sum().item()
        actual_labels.extend(data.y.int().tolist())
        predicted_labels.extend(pred.tolist())
    
    accuracy = correct/len(test_data)

    print(f'Accuracy: {accuracy:.4f}')

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for true, predicted in zip(actual_labels, predicted_labels):
        if predicted == 1 and true == 1:
            true_positives += 1
        elif predicted == 1 and true == 0:
            false_positives += 1
        elif predicted == 0 and true == 0:
            true_negatives += 1
        elif predicted == 0 and true == 1:
            false_negatives += 1

    conf_matrix = np.array([[true_positives, false_negatives], [false_positives, true_negatives]])
    labels = ['Positive', 'Negative']
    categories = ['Positive', 'Negative']

    plt.figure(figsize=(6, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    precision = true_positives/(true_positives + false_positives)
    print(f'Precision: {precision:.4f}')
    
    recall = true_positives/(true_positives + false_negatives)
    print(f'Recall: {recall:.4f}')
    
    f1_score = (2 * precision * recall) / (precision + recall)
    print(f'F1 Score: {f1_score:.4f}')



if __name__ == '__main__':
    main()
