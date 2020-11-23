import os

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

from random_translate import *

detoken = TreebankWordDetokenizer()


if __name__ == "__main__":
    full_df = pd.read_csv("val_data.csv", encoding="utf-8", index_col=0)
    eval_df = pd.read_csv("eval_data_only.csv", encoding="utf-8", index_col=0)
    print("Shapes are ", full_df.shape, eval_df.shape)

    print("Cleaning text")
    eval_df["original"] = eval_df["original"].apply(lambda x: x.replace("<unk>", "").replace("<blank>", ""))
    eval_df["edited"] = eval_df["edited"].apply(lambda x: x.replace("<unk>", "").replace("<blank>", ""))
    print("Finding matches")
    eval_df["original"] = eval_df["original"].apply(lambda x: process.extractOne(x, full_df["original"])[0])
    print("Finding Edited matches")
    eval_df["edited"] = eval_df["edited"].apply(lambda x: process.extractOne(x, full_df["edited"])[0])
    print("Finding rows that dont match still")
    to_drop_rows = []
    # some didn't find matches because of the <unk>
    for index, row in eval_df.iterrows():
        if not (full_df[["original", "edited"]] == row[["original", "edited"]]).all(1).any():
            to_drop_rows.append(index)
    eval_df = eval_df.drop(eval_df.index[to_drop_rows])
    eval_df.to_csv("unpunct.csv")
    # clean up punctuation
    eval_df["original"] = eval_df["original"].apply(lambda x: detoken.detokenize(x.split(" ")))
    eval_df["translated"] = eval_df["translated"].apply(lambda x: detoken.detokenize(x.split(" ")))
    eval_df["edited"] = eval_df["edited"].apply(lambda x: detoken.detokenize(x.split(" ")))
    # add in random model
    eval_df["random"] = eval_df["original"]
    eval_df["random"] = eval_df["random"].apply(lambda x: random_translate(x))
    eval_df.to_csv("ready_for_mturk.csv")


