# networkify.py
# Author: Wyatt McMullen
#
# Script for generating edge lists for use in generating networks of brain connectomes.

"""
Networkify

This is the primary script for transforming edge lists of processed fMRI data from subjects into 
adjacency lists providing methods to construct these structures for analysis of subnetworks in brain connectomes.
"""

#Imports
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
import time

def to_connectome(test_path,out_path):
    if not os.path.exists(test_path):
        print('Invalid input file path.')
        return

    files = os.listdir(test_path)

    for filename in tqdm(files, desc="Making Matrices"):
        file_path = os.path.join(test_path, filename)

        with open(file_path, 'r') as f:
            edges = pd.read_csv(f, sep=',', index_col=[0]) 
            edges = edges.melt(ignore_index=False).reset_index()
            edges.rename(columns={'index':'regionA','variable':'regionB','value':'Corr'},inplace=True)


            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)

            edges.to_csv(os.path.join(out_path, "ed_" + filename))
        
        time.sleep(0.01)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description=__doc__)

    #Arguments

    #Required Args
    parser.add_argument('-i','--input_dir',required=True,nargs=1,type=str,help='Path to local folder containing .1D files to convert.') #Local Input Path
    parser.add_argument('-o','--output_dir',required=True,nargs=1,type=str,help='Path to local folder to store created .csv files.') #Local Output Path

    #Optional Args / Prospective Args


    args = parser.parse_args()

    to_connectome(os.path.abspath(args.input_dir[0]),os.path.abspath(args.output_dir[0]))