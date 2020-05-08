import argparse
import logging
import pickle
import sklearn
# import pandas as pd
import numpy as np
import os
import csv


def load_data(data_dir):
    with open(os.path.join(data_dir, "train.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        train_data = [line for line in reader]

    with open(os.path.join(data_dir, "valid.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        test_data = [line for line in reader]

    with open(os.path.join(data_dir, "test.csv"), "r") as f:
        reader = csv.reader(f, delimiter="\t")
        test_data = [line for line in reader]
    


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    data_dir = "./data"
    
    load_data(data_dir)

if __name__ == "__main__":
    main()
