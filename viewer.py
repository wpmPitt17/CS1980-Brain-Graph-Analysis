import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Path directory for all test and output files
test_path = './test_files/rois_cc200'
out_path = './output_correlation'

for filename in os.listdir(test_path):
    file_path = os.path.join(test_path, filename)

    with open(file_path, 'r') as f:
        df = pd.read_csv(f, sep='\s+') 
        new_corr = df.corr(method='spearman')

        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

        new_corr.to_csv(os.path.join(out_path, filename))

adjacency = df.corr(method='spearman')
plt.figure(figsize=(10, 8))

sns.heatmap(data=new_corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.title(f"Spearman Correlation Heatmap for Last")

plt.show()