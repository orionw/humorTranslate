import os
import re
import csv

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_files():
    """
    Takes the original Humicroedit dataset and creates a file with unique sentences for processing
    """
    train = pd.read_csv(os.path.join("data", "train.csv"))
    dev = pd.read_csv(os.path.join("data", "dev.csv"))

    ### FOR HumorNMT ###
    full = pd.concat([train, dev])

    edited = []
    original = []
    for index, (row) in full.iterrows():
        edit = re.sub(r'\<.*?/\>', row["edit"], row["original"])
        edited.append(edit)
        assert "\<" not in edit, "did not replace, error"
        original.append(row["original"].replace("<", "").replace("/>", ""))

    full["edited_version"] = edited
    full["original_clean"] = original

    nmt = full.drop_duplicates(subset=["original"])
    print("Unique sentences:", nmt.shape[0])
    nmt.to_csv(os.path.join("data", "full_unique.csv"))


if __name__ == "__main__":
    create_files()